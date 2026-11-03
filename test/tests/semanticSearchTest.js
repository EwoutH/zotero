describe("Zotero.SemanticSearch", function () {
	var embedStub;
	var libraryID;
	
	// A stand-in for the model: words hashed into buckets, so texts sharing words are similar
	function fakeEmbed(texts) {
		return texts.map((text) => {
			let vector = new Float32Array(256);
			for (let word of text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []) {
				let hash = 0;
				for (let char of word) {
					hash = (hash * 31 + char.codePointAt(0)) % 256;
				}
				vector[hash] += 1;
			}
			let norm = Math.hypot(...vector) || 1;
			return vector.map(x => x / norm);
		});
	}
	
	async function createIndexedItem(title, abstract = '') {
		let item = createUnsavedDataObject('item', { title });
		item.setField('abstractNote', abstract);
		await item.saveTx();
		await Zotero.SemanticSearch.index();
		return item;
	}
	
	before(async function () {
		libraryID = Zotero.Libraries.userLibraryID;
		embedStub = sinon.stub(Zotero.SemanticSearch, 'embed').callsFake(async texts => fakeEmbed(texts));
		Zotero.Prefs.set('search.semantic.enabled', true);
		await Zotero.SemanticSearch.start();
		await Zotero.SemanticSearch.index();
	});
	
	after(function () {
		Zotero.Prefs.clear('search.semantic.enabled');
		embedStub.restore();
	});
	
	describe("#fuse()", function () {
		it("should rank an item found by several rankings above one found by one", function () {
			let results = Zotero.SemanticSearch.fuse({
				semantic: [1, 2, 3],
				metadata: [3, 4]
			});
			assert.equal(results[0].itemID, 3);
			assert.deepEqual(results[0].ranks, { semantic: 3, metadata: 1 });
			assert.sameMembers(results.map(r => r.itemID), [1, 2, 3, 4]);
		});
	});
	
	describe("#getMatchExpression()", function () {
		it("should OR the query's distinct words as FTS5 strings", function () {
			assert.equal(
				Zotero.SemanticSearch.getMatchExpression('river "flood" models, river*'),
				'"river" OR "flood" OR "models"'
			);
		});
		
		it("should return an empty expression for a query without words", function () {
			assert.equal(Zotero.SemanticSearch.getMatchExpression('"* -'), '');
		});
	});
	
	describe("#search()", function () {
		it("should rank an item by its title and abstract", async function () {
			let match = await createIndexedItem('Quokka habitats', 'Marsupials on Rottnest Island');
			await createIndexedItem('Unrelated title about compilers');
			
			let results = await Zotero.SemanticSearch.search('quokka marsupials', { libraryID });
			assert.equal(results[0].itemID, match.id);
			assert.isDefined(results[0].ranks.semantic);
			assert.isDefined(results[0].ranks.metadata);
		});
		
		it("should reindex an item when it's modified", async function () {
			let item = await createIndexedItem('Wombat burrows');
			item.setField('title', 'Platypus venom');
			await item.saveTx();
			await Zotero.SemanticSearch.index();
			
			let results = await Zotero.SemanticSearch.search('platypus', { libraryID });
			assert.equal(results[0].itemID, item.id);
			results = await Zotero.SemanticSearch.search('wombat burrows', { libraryID });
			assert.notInclude(results.map(r => r.itemID), item.id);
		});
		
		it("should leave out trashed items", async function () {
			let item = await createIndexedItem('Numbat termite diet');
			item.deleted = true;
			await item.saveTx();
			
			let results = await Zotero.SemanticSearch.search('numbat termite', { libraryID });
			assert.notInclude(results.map(r => r.itemID), item.id);
		});
		
		it("should drop an erased item from the index", async function () {
			let item = await createIndexedItem('Bilby conservation');
			let id = item.id;
			await item.eraseTx();
			
			let rows = await Zotero.DB.valueQueryAsync(
				"SELECT COUNT(*) FROM semantic.itemVectors WHERE itemID=?", id
			);
			assert.equal(rows, 0);
		});
		
		it("should limit results to the given items", async function () {
			let a = await createIndexedItem('Echidna spines');
			let b = await createIndexedItem('Echidna eggs');
			
			let results = await Zotero.SemanticSearch.search('echidna', { libraryID, itemIDs: [b.id] });
			assert.deepEqual(results.map(r => r.itemID), [b.id]);
			assert.notInclude(results.map(r => r.itemID), a.id);
		});
		
		it("should return nothing when disabled", async function () {
			await createIndexedItem('Dingo howls');
			Zotero.Prefs.set('search.semantic.enabled', false);
			try {
				assert.lengthOf(await Zotero.SemanticSearch.search('dingo', { libraryID }), 0);
			}
			finally {
				Zotero.Prefs.set('search.semantic.enabled', true);
			}
		});
	});
});
