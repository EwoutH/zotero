/*
    ***** BEGIN LICENSE BLOCK *****

    Copyright © 2026 Corporation for Digital Scholarship
                     Vienna, Virginia, USA
                     https://www.zotero.org

    This file is part of Zotero.

    Zotero is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Zotero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Zotero.  If not, see <http://www.gnu.org/licenses/>.

    ***** END LICENSE BLOCK *****
*/

/**
 * Zotero.SemanticSearch -- ranked search over item metadata, by meaning and by words.
 *
 * Each regular item's title, abstract and tags are embedded with a static embedding model
 * (Mozilla's copy of potion-multilingual-128M, run by Firefox's ML runtime). A static model is a
 * token -> vector table that gets averaged, so embedding costs well under a millisecond per item
 * and a whole library indexes in seconds, with no indexing machinery beyond a notifier queue.
 * The same text also goes into a small FTS5 table for BM25.
 *
 * search() ranks a library three ways -- vector similarity of the metadata, BM25 of the
 * metadata, and BM25 of attachment full text (from the existing full-text index) -- and fuses
 * the rankings with Reciprocal Rank Fusion.
 *
 * Vectors live in an attached database (semantic.sqlite) that's derived and rebuildable, like
 * the full-text index, and is recreated whenever the model or the text recipe changes.
 */
Zotero.SemanticSearch = new function () {
	const MODEL = Object.freeze({
		modelId: 'mozilla/static-embeddings',
		revision: 'v1.0.0',
		subfolder: 'models/minishlab/potion-multilingual-128M',
		dtype: 'fp8_e4m3',
		dims: 256
	});
	// Bump when what's embedded for an item changes
	const TEXT_VERSION = 1;
	// Bump when the schema changes
	const DB_VERSION = 1;
	// Everything that would change the stored vectors
	const INDEX_VERSION = [MODEL.subfolder, MODEL.dtype, MODEL.dims, TEXT_VERSION, DB_VERSION].join('/');
	
	const BATCH_SIZE = 200;
	const INDEX_DELAY = 2000;
	// How deep each ranking goes before fusion
	const CANDIDATES = 200;
	// Cosine distance past which an item isn't a semantic match at all. Firefox's semantic
	// history search uses 0.6 with a similar static model; not yet tuned for this one.
	const MAX_DISTANCE = 0.6;
	// The constant in an RRF contribution, 1 / (RRF_K + rank), as in the original paper
	const RRF_K = 60;
	
	let _dbReady = null;
	let _observerID = null;
	let _engine = null;
	let _runtimeConfigured = false;
	let _pendingIDs = new Set();
	let _indexTimer = null;
	let _indexing = null;
	
	this.isEnabled = function () {
		return !!Zotero.Prefs.get('search.semantic.enabled')
			&& Services.prefs.getBoolPref('browser.ml.enable', false);
	};
	
	this.init = async function () {
		Zotero.Prefs.registerObserver('search.semantic.enabled', () => {
			if (this.isEnabled()) {
				this.start().catch(e => Zotero.logError(e));
			}
			else {
				this.stop();
			}
		});
		if (this.isEnabled()) {
			await this.start();
		}
	};
	
	/**
	 * Attach the index and start keeping it up to date
	 *
	 * @return {Promise}
	 */
	this.start = async function () {
		if (!_dbReady) {
			_dbReady = (async () => {
				await Zotero.DB.loadExtension('vec');
				await _setUpDB();
				// An attached database doesn't survive a reconnect (e.g., after a vacuum)
				Zotero.DB.onConnect(_setUpDB);
			})();
			_dbReady.catch(() => {
				_dbReady = null;
			});
		}
		await _dbReady;
		if (!_observerID) {
			_observerID = Zotero.Notifier.registerObserver(this, ['item'], 'semanticSearch');
			// Catch up on whatever isn't indexed yet
			_scheduleIndexing();
		}
	};
	
	/**
	 * Stop indexing. What's indexed is kept, so starting again resumes.
	 */
	this.stop = function () {
		if (_observerID) {
			Zotero.Notifier.unregisterObserver(_observerID);
			_observerID = null;
		}
		clearTimeout(_indexTimer);
		_pendingIDs.clear();
	};
	
	async function _setUpDB() {
		let attached = (await Zotero.DB.queryAsync("PRAGMA database_list"))
			.some(row => row.name == 'semantic');
		if (!attached) {
			let path = Zotero.DataDirectory.getDatabase('semantic');
			await Zotero.DB.queryAsync("ATTACH DATABASE ? AS semantic", [path]);
		}
		// itemIDs are reassigned when zotero.sqlite is recreated, and vectors from another model
		// or text recipe can't be compared with this one's, so either means starting over
		let stamp = INDEX_VERSION + '/' + Zotero.Users.getLocalUserKey();
		let stored = await Zotero.DB.tableExists('meta', 'semantic')
			&& await Zotero.DB.valueQueryAsync("SELECT value FROM semantic.meta WHERE key='stamp'");
		if (stored === stamp) {
			return;
		}
		Zotero.debug("SemanticSearch: creating index");
		for (let table of ['meta', 'itemVectors', 'itemText']) {
			await Zotero.DB.queryAsync(`DROP TABLE IF EXISTS semantic.${table}`);
		}
		// A float32 vector per item, or NULL for an item with no text, so that it isn't
		// picked up again by the catch-up query
		await Zotero.DB.queryAsync("CREATE TABLE semantic.itemVectors "
			+ "(itemID INTEGER PRIMARY KEY, embedding BLOB)");
		// The same text, for BM25 over the metadata
		await Zotero.DB.queryAsync("CREATE VIRTUAL TABLE semantic.itemText USING fts5("
			+ "text, tokenize='unicode61 remove_diacritics 2', content='', contentless_delete=1)");
		await Zotero.DB.queryAsync("CREATE TABLE semantic.meta (key TEXT PRIMARY KEY, value)");
		await Zotero.DB.queryAsync("INSERT INTO semantic.meta VALUES ('stamp', ?)", [stamp]);
	}
	
	//
	// Indexing
	//
	
	this.notify = async function (action, type, ids) {
		if (type != 'item' || !_observerID) {
			return;
		}
		if (action == 'add' || action == 'modify') {
			for (let id of ids) {
				_pendingIDs.add(id);
			}
			_scheduleIndexing();
		}
		else if (action == 'delete') {
			await _deleteRows(ids);
		}
	};
	
	function _scheduleIndexing() {
		clearTimeout(_indexTimer);
		_indexTimer = setTimeout(() => {
			Zotero.SemanticSearch.index().catch(e => Zotero.logError(e));
		}, INDEX_DELAY);
	}
	
	/**
	 * Embed every regular item that's changed or not indexed yet
	 *
	 * @return {Promise}
	 */
	this.index = function () {
		// One pass at a time: a call during a pass runs another after it, for what arrived meanwhile
		_indexing = (_indexing || Promise.resolve()).catch(() => {}).then(_indexAll);
		return _indexing;
	};
	
	async function _indexAll() {
		let start = Date.now();
		let count = 0;
		// Until stopped or caught up
		while (true) {
			if (!_observerID) {
				break;
			}
			let ids = [..._pendingIDs].slice(0, BATCH_SIZE);
			ids.forEach(id => _pendingIDs.delete(id));
			if (!ids.length) {
				ids = await _getUnindexedItemIDs(BATCH_SIZE);
			}
			if (!ids.length) {
				break;
			}
			await _indexItems(ids);
			count += ids.length;
		}
		if (count) {
			Zotero.debug(`SemanticSearch: indexed ${count} items in ${Date.now() - start} ms`);
		}
	}
	
	function _getUnindexedItemIDs(limit) {
		return Zotero.DB.columnQueryAsync(
			"SELECT itemID FROM items JOIN libraries USING (libraryID) "
				+ "WHERE libraries.type != 'feed' "
				+ "AND itemTypeID NOT IN (?, ?, ?) "
				+ "AND itemID NOT IN (SELECT itemID FROM semantic.itemVectors) "
				+ "LIMIT ?",
			[...['note', 'attachment', 'annotation'].map(type => Zotero.ItemTypes.getID(type)), limit]
		);
	}
	
	async function _indexItems(ids) {
		let items = (await Zotero.Items.getAsync(ids))
			.filter(item => item && item.isRegularItem() && !item.isFeedItem);
		await Zotero.Items.loadDataTypes(items, ['itemData', 'tags']);
		let texts = items.map(item => Zotero.SemanticSearch.getItemText(item));
		let vectors = await Zotero.SemanticSearch.embed(texts.filter(Boolean));
		await Zotero.DB.executeTransaction(async function () {
			await _deleteRows(items.map(item => item.id));
			let v = 0;
			for (let i = 0; i < items.length; i++) {
				if (!texts[i]) {
					await Zotero.DB.queryAsync(
						"INSERT INTO semantic.itemVectors (itemID, embedding) VALUES (?, NULL)",
						[items[i].id]
					);
				}
				else {
					// Bound as JSON text, since mozStorage doesn't bind typed arrays as BLOBs
					await Zotero.DB.queryAsync(
						"INSERT INTO semantic.itemVectors (itemID, embedding) VALUES (?, vec_f32(?))",
						[items[i].id, JSON.stringify(Array.from(vectors[v++]))]
					);
					await Zotero.DB.queryAsync(
						"INSERT INTO semantic.itemText (rowid, text) VALUES (?, ?)",
						[items[i].id, texts[i]]
					);
				}
			}
		});
	}
	
	async function _deleteRows(ids) {
		if (!ids.length) {
			return;
		}
		await Zotero.Utilities.Internal.forEachChunkAsync(ids, Zotero.DB.MAX_BOUND_PARAMETERS, async (chunk) => {
			let placeholders = chunk.map(() => '?').join(', ');
			await Zotero.DB.queryAsync(
				`DELETE FROM semantic.itemVectors WHERE itemID IN (${placeholders})`, chunk
			);
			await Zotero.DB.queryAsync(
				`DELETE FROM semantic.itemText WHERE rowid IN (${placeholders})`, chunk
			);
		});
	}
	
	/**
	 * The text embedded for an item: title, abstract and tags
	 *
	 * @param {Zotero.Item} item
	 * @return {String}
	 */
	this.getItemText = function (item) {
		return [
			item.getField('title', false, true),
			item.getField('abstractNote'),
			item.getTags().map(tag => tag.tag).join(', ')
		].filter(Boolean).join('\n');
	};
	
	//
	// Embedding
	//
	
	/**
	 * Embed texts as unit vectors
	 *
	 * @param {String[]} texts
	 * @return {Promise<Float32Array[]>}
	 */
	this.embed = async function (texts) {
		if (!texts.length) {
			return [];
		}
		let engine = await _getEngine();
		let { output } = await engine.run({
			args: texts.map(text => text.replace(/\s+/g, ' ').trim()),
			options: { pooling: 'mean', normalize: true }
		});
		return output;
	};
	
	async function _getEngine() {
		// The runtime shuts an idle engine down on its own timer
		let engine = _engine && await _engine;
		if (engine && ['closed', 'crashed', 'error'].includes(engine.engineStatus)) {
			_engine = null;
		}
		if (!_engine) {
			_engine = _createEngine();
			_engine.catch(() => {
				_engine = null;
			});
		}
		return _engine;
	}
	
	function _createEngine() {
		_configureRuntime();
		let { createEngine } = ChromeUtils.importESModule(
			"chrome://global/content/ml/EngineProcess.sys.mjs"
		);
		// Downloads the model on first use; the runtime caches it
		return createEngine({
			engineId: 'zotero-semantic-search',
			taskName: 'static-embeddings',
			backend: 'static-embeddings',
			modelId: MODEL.modelId,
			modelRevision: MODEL.revision,
			modelHubRootUrl: 'https://model-hub.mozilla.org',
			modelHubUrlTemplate: '{model}/{revision}',
			staticEmbeddingsOptions: {
				subfolder: MODEL.subfolder,
				dtype: MODEL.dtype,
				dimensions: MODEL.dims,
				compression: true
			}
		});
	}
	
	// The ML runtime reads its configuration and its model-host allow list from Remote Settings,
	// which Zotero doesn't include. Supply empty configuration (it then uses what createEngine()
	// is given) and an allow list with Mozilla's model hub.
	function _configureRuntime() {
		if (_runtimeConfigured) {
			return;
		}
		let { MLEngineParent } = ChromeUtils.importESModule(
			"resource://gre/actors/MLEngineParent.sys.mjs"
		);
		let client = records => ({
			get: async () => records,
			on() {},
			off() {}
		});
		MLEngineParent.mockRemoteSettings({
			'ml-onnx-runtime': client([]),
			'ml-inference-options': client([]),
			'ml-model-allow-deny-list': client([
				{ filter: 'ALLOW', urlPrefix: 'https://model-hub.mozilla.org/' }
			])
		});
		_runtimeConfigured = true;
	}
	
	//
	// Search
	//
	
	/**
	 * Rank a library's regular items against a query
	 *
	 * @param {String} query
	 * @param {Object} options
	 * @param {Number} options.libraryID
	 * @param {Number[]} [options.itemIDs] - Limit results to these items (e.g., a collection)
	 * @param {Number} [options.limit=100]
	 * @return {Promise<Object[]>} - [{ itemID, score, ranks: { semantic, metadata, fulltext } }],
	 *     best first. Ranks are 1-based, and missing for a ranking that didn't include the item.
	 */
	this.search = async function (query, { libraryID, itemIDs, limit = 100 }) {
		query = query.trim();
		if (!query || !this.isEnabled()) {
			return [];
		}
		await this.start();
		let [semantic, metadata, fulltext] = await Promise.all([
			_rankBySimilarity(query, libraryID),
			_rankByMetadataBM25(query, libraryID),
			_rankByFulltextBM25(query, libraryID)
		]);
		let results = this.fuse({ semantic, metadata, fulltext });
		if (itemIDs) {
			let scope = new Set(itemIDs);
			results = results.filter(result => scope.has(result.itemID));
		}
		return results.slice(0, limit);
	};
	
	/**
	 * Reciprocal Rank Fusion: each ranking adds 1 / (RRF_K + rank) to an item's score
	 *
	 * @param {Object<String, Number[]>} rankings - Item IDs, best first, by ranking name
	 * @return {Object[]} - [{ itemID, score, ranks }], best first
	 */
	this.fuse = function (rankings) {
		let results = new Map();
		for (let [name, ids] of Object.entries(rankings)) {
			ids.forEach((itemID, i) => {
				let result = results.get(itemID);
				if (!result) {
					result = { itemID, score: 0, ranks: {} };
					results.set(itemID, result);
				}
				result.score += 1 / (RRF_K + i + 1);
				result.ranks[name] = i + 1;
			});
		}
		return [...results.values()].sort((a, b) => b.score - a.score);
	};
	
	/**
	 * A query as an FTS5 expression matching any of its words
	 *
	 * @param {String} query
	 * @return {String}
	 */
	this.getMatchExpression = function (query) {
		let words = query.match(/[\p{L}\p{N}]+/gu) || [];
		return [...new Set(words)].map(word => `"${word}"`).join(' OR ');
	};
	
	async function _rankBySimilarity(query, libraryID) {
		let [vector] = await Zotero.SemanticSearch.embed([query]);
		return Zotero.DB.columnQueryAsync(
			"SELECT itemID FROM ("
				+ "SELECT itemID, vec_distance_cosine(embedding, vec_f32(?)) AS distance "
				+ "FROM semantic.itemVectors JOIN items USING (itemID) "
				+ "WHERE libraryID=? AND embedding IS NOT NULL "
				+ "AND itemID NOT IN (SELECT itemID FROM deletedItems)"
				+ ") WHERE distance < ? ORDER BY distance LIMIT ?",
			[JSON.stringify(Array.from(vector)), libraryID, MAX_DISTANCE, CANDIDATES]
		);
	}
	
	async function _rankByMetadataBM25(query, libraryID) {
		let expression = Zotero.SemanticSearch.getMatchExpression(query);
		if (!expression) {
			return [];
		}
		return Zotero.DB.columnQueryAsync(
			"SELECT T.rowid FROM semantic.itemText T JOIN items I ON (I.itemID = T.rowid) "
				+ "WHERE itemText MATCH ? AND I.libraryID=? "
				+ "AND T.rowid NOT IN (SELECT itemID FROM deletedItems) "
				+ "ORDER BY bm25(itemText) LIMIT ?",
			[expression, libraryID, CANDIDATES]
		);
	}
	
	// BM25 over attachment full text, from the existing full-text index, with an attachment
	// standing in for its parent item
	async function _rankByFulltextBM25(query, libraryID) {
		let expression = Zotero.SemanticSearch.getMatchExpression(query);
		if (!expression) {
			return [];
		}
		let ids = await Zotero.DB.columnQueryAsync(
			"SELECT COALESCE(IA.parentItemID, IA.itemID) FROM ("
				+ "SELECT rowid, bm25(fulltextContent) AS score FROM ftindex.fulltextContent "
				+ "WHERE fulltextContent MATCH ?"
				+ ") C JOIN itemAttachments IA ON (IA.itemID = C.rowid) "
				+ "JOIN items I ON (I.itemID = IA.itemID) "
				+ "WHERE I.libraryID=? AND IA.itemID NOT IN (SELECT itemID FROM deletedItems) "
				+ "ORDER BY C.score LIMIT ?",
			[expression, libraryID, CANDIDATES * 2]
		);
		// An item ranks by its best attachment
		return [...new Set(ids)].slice(0, CANDIDATES);
	}
};
