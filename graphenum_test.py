import pytest
from easie.building_blocks.file_pricer import FilePricer
from easie.graphenum import RxnGraphEnumerator
from easie.plausibility.fast_filter import FastFilterScorer
from rxnmapper import RXNMapper
from rdkit import Chem
from rdkit.Chem.FilterCatalog import *

class TestRxnGraphEnumerator:
    def setup_method(self):
        self.pricer = FilePricer()
        self.pricer.load(
            path="easie/building_blocks/buyables_test.json.gz", precompute_mols=True
        )
        rxn_mapper = RXNMapper()
        self.mapper = lambda x : rxn_mapper.get_attention_guided_atom_maps([x])[0]['mapped_rxn']
        
        model = FastFilterScorer()
        model.load('easie/plausibility/fast_filter/1')
        self.fast_filter = model.predict
        
    def test_diphenhydramine_unmapped_1(self):
        reaction_smiles = [
            "BrC(c1ccccc1)c2ccccc2.CN(CCO)C>>CN(CCOC(c3ccccc3)c4ccccc4)C"
        ]
        graph = RxnGraphEnumerator(reaction_smiles, mapper=self.mapper, forward_filter=self.fast_filter)
        graph.search_building_blocks(self.pricer)

        combos = graph.count_combinations()
        library = set([d["smiles"] for d in graph.library_generator()])
        expected = set(
            [
                "CN(C)CCCOC(c1ccccc1)c1ccccc1",
                "CN(C)CCCOC(c1ccc(O)cc1)c1ccc(O)cc1",
                "CN(C)CCOC(c1ccccc1)c1ccccc1",
                "CN(C)CCOC(c1ccc(O)cc1)c1ccc(O)cc1",
                "CC(COC(c1ccccc1)c1ccccc1)N(C)C",
                "CC(COC(c1ccc(O)cc1)c1ccc(O)cc1)N(C)C",
                "c1ccc(C(OC(c2ccccc2)c2ccccc2)c2ccccc2)cc1",
                "Oc1ccc(C(OC(c2ccc(F)cc2)c2ccc(F)cc2)c2ccc(O)cc2)cc1",
                "Fc1ccc(C(OC(c2ccccc2)c2ccccc2)c2ccc(F)cc2)cc1",
                "Oc1ccc(C(OC(c2ccccc2)c2ccccc2)c2ccc(O)cc2)cc1",
            ]
        )

        assert library == expected
        assert combos == 10
        assert len(library) == combos

        graph.filter_by_similarity(threshold=0.3)
        graph.filter_by_plausibility(threshold=0.1)

        filtered_combos = graph.count_combinations()
        filtered_library = set([d["smiles"] for d in graph.library_generator()])
        assert filtered_combos <= combos
        assert len(filtered_library) == filtered_combos

    def test_diphenhydramine_unmapped_2(self):
        reaction_smiles = [
            "BrCCOC(c1ccccc1)c1ccccc1.CNC>>CN(C)CCOC(c1ccccc1)c1ccccc1",
            "BrCCBr.OC(c1ccccc1)c1ccccc1>>BrCCOC(c1ccccc1)c1ccccc1",
        ]
        graph = RxnGraphEnumerator(reaction_smiles, mapper=self.mapper, forward_filter=self.fast_filter)
        graph.search_building_blocks(self.pricer)
        graph.filter_by_similarity(threshold=0.41)

        combos = graph.count_combinations()
        library = set([d["smiles"] for d in graph.library_generator()])
        expected = set(
            ['CN(C)CCCOC(c1ccccc1)c1ccccc1', 'CN(C)CCOC(c1ccccc1)c1ccccc1', 'CN(C)CCOC(c1ccc(F)cc1)c1ccc(F)cc1', 'CN(C)CCCOC(c1ccc(F)cc1)c1ccc(F)cc1']
        )
        print (list(library))
        print (list(expected))
        assert library == expected
        assert combos == 4
        assert len(library) == combos

        graph.filter_by_similarity(threshold=0.3)
        graph.filter_by_plausibility(threshold=0.1)

        filtered_combos = graph.count_combinations()
        filtered_library = set([d["smiles"] for d in graph.library_generator()])
        assert filtered_combos <= combos
        assert len(filtered_library) == filtered_combos

    def test_diphenhydramine_protection_1(self):
        reaction_smiles = [
            "[Br:1][CH2:2][CH2:3][Br:18].[OH:4][CH:5]([c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1)[c:12]1[cH:13][cH:14][cH:15][cH:16][cH:17]1>>[Br:1][CH2:2][CH2:3][O:4][CH:5]([c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1)[c:12]1[cH:13][cH:14][cH:15][cH:16][cH:17]1",
            "[Br:1][CH2:2][CH2:3][O:4][CH:5]([c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1)[c:12]1[cH:13][cH:14][cH:15][cH:16][cH:17]1.[OH:18][c:19]1[cH:20][cH:21][cH:22][cH:23][cH:24]1>>[CH2:2]([CH2:3][O:4][CH:5]([c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1)[c:12]1[cH:13][cH:14][cH:15][cH:16][cH:17]1)[O:18][c:19]1[cH:20][cH:21][cH:22][cH:23][cH:24]1",
            "[CH3:1][NH:2][CH3:3].[cH:4]1[cH:5][cH:6][c:7]([O:8][CH2:9][CH2:10][O:11][CH:12]([c:13]2[cH:14][cH:15][cH:16][cH:17][cH:18]2)[c:19]2[cH:20][cH:21][cH:22][cH:23][cH:24]2)[cH:25][cH:26]1>>[CH3:1][N:2]([CH3:3])[CH2:9][CH2:10][O:11][CH:12]([c:13]1[cH:14][cH:15][cH:16][cH:17][cH:18]1)[c:19]1[cH:20][cH:21][cH:22][cH:23][cH:24]1",
        ]
        graph = RxnGraphEnumerator(reaction_smiles, mapper=self.mapper)
        graph.search_building_blocks(self.pricer)
        graph.filter_by_similarity(threshold=0.21)

        all_combos = graph.count_combinations(deduplicate=False)
        deduped_combos = graph.count_combinations()
        library = set(
            [d["smiles"] for d in graph.library_generator() if len(d["smiles"])]
        )
        expected = set(
            ['CN(C)CCCOC(c1ccccc1)c1ccccc1', 'CN(C)CCOC(c1ccccc1)c1ccccc1', 'CN(C)CCOC(c1ccc(F)cc1)c1ccc(F)cc1', 'CN(C)CCCOC(c1ccc(F)cc1)c1ccc(F)cc1']
        )

        assert all_combos == 8
        assert deduped_combos == 4
        assert library == expected

    def test_diphenhydramine_parallel_5(self):
        reaction_smiles = [
            "BrC(c1ccccc1)c2ccccc2.CN(CCO)C>>CN(CCOC(c3ccccc3)c4ccccc4)C"
        ]
        graph = RxnGraphEnumerator(reaction_smiles, mapper=self.mapper)
        graph.search_building_blocks(self.pricer)

        combos = graph.count_combinations()

        result = graph.generate_library(nproc=8)
        assert combos == len(result)

    def test_mitapivat_graph_construction_1(self):
        reaction_smiles = [
            "CS(=O)(=O)OCCN(CCNCC1CC1)C(=O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1>>O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1",
            "CS(=O)(=O)Cl.O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N(CCO)CCNCC1CC1>>CS(=O)(=O)OCCN(CCNCC1CC1)C(=O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1",
            "O=C(Cl)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1.OCCNCCNCC1CC1>>O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N(CCO)CCNCC1CC1",
            "O=C(O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1>>O=C(Cl)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1",
            "ClCC1CC1.NCCNCCO>>OCCNCCNCC1CC1",
        ]
        reaction_smarts = [
            "[C:2]-[CH2;D2;+0:1]-[N;H0;D3;+0:4](-[C:3])-[C:5]>>C-S(=O)(=O)-O-[CH2;D2;+0:1]-[C:2].[C:3]-[NH;D2;+0:4]-[C:5]",
            "[C:5]-[O;H0;D2;+0:6]-[S;H0;D4;+0:1](-[C;D1;H3:2])(=[O;D1;H0:3])=[O;D1;H0:4]>>Cl-[S;H0;D4;+0:1](-[C;D1;H3:2])(=[O;D1;H0:3])=[O;D1;H0:4].[C:5]-[OH;D1;+0:6]",
            "[C:4]-[N;H0;D3;+0:5](-[C:6])-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3]>>Cl-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3].[C:4]-[NH;D2;+0:5]-[C:6]",
            "[Cl;H0;D1;+0]-[C;H0;D3;+0:1](=[O;H0;D1;+0:2])-[c:3]>>O=[C;H0;D3;+0:1](-[OH;D1;+0:2])-[c:3]",
            "[C:3]-[NH;D2;+0:4]-[CH2;D2;+0:1]-[C:2]>>Cl-[CH2;D2;+0:1]-[C:2].[C:3]-[NH2;D1;+0:4]",
        ]
        graph = RxnGraphEnumerator(reaction_smiles, reaction_smarts=reaction_smarts)
        for n in graph.graph.nodes:
            if graph.graph.nodes[n]["type"] == "chemical":
                orig_smiles = graph.graph.nodes[n]["smiles"]
                isotope_smiles = graph.bb_labeled_graph.nodes[n]["smiles"]
                m = Chem.MolFromSmiles(isotope_smiles)
                [a.SetIsotope(0) for a in m.GetAtoms()]

                assert Chem.MolToSmiles(m) == orig_smiles

    def test_tapinarof_query_definition (self):
        route_1 = [
            'CC(C)c1c(O)cc(C=O)cc1O.O=C(O)Cc1ccccc1>>CC(C)c1c(O)cc(/C=C/c2ccccc2)cc1O', 
            'CC(C)c1c(O)cccc1O>>CC(C)c1c(O)cc(C=O)cc1O',
            'CC(C)c1ccccc1O>>CC(C)c1c(O)cccc1O'
        ]
        graph_1 = RxnGraphEnumerator(route_1, mapper=self.mapper)
        print ([graph_1.graph[n] for n in graph_1.reaction_nodes])
        for l in graph_1.leaves:
            if l['smiles'] == 'CC(C)c1ccccc1O':
                print (l['query'])
                # assert len(l['query'].split('.')) == 2
            elif l['smiles'] == 'O=C(O)Cc1ccccc1':
                assert len(l['query'].split('.')) == 1

        route_2 = [
            'CC(C)c1c(O)cc(C=O)cc1O.O=C(O)Cc1ccccc1>>CC(C)c1c(O)cc(/C=C/c2ccccc2)cc1O', 
            'C1N2CN3CN1CN(C2)C3.CC(C)c1c(O)cccc1O>>CC(C)c1c(O)cc(C=O)cc1O',
            'CC(C)c1ccccc1O>>CC(C)c1c(O)cccc1O'
        ]
        graph_2 = RxnGraphEnumerator(route_2, mapper=self.mapper)
        print ([graph_2.graph[n] for n in graph_2.reaction_nodes])
        for l_1 in graph_1.leaves:
            for l_2 in graph_2.leaves:
                if l_1['smiles'] == l_2['smiles']:
                    assert l_1['query'] == l_2['query']

    def test_ZINC206382 (self):
        route = [
            'c1ccc(C[O:1][c:2]2[cH:3][cH:4][c:5]([NH:6][c:7]3[n:8][c:9](-[c:10]4[cH:11][cH:12][c:13]([Cl:14])[cH:15][cH:16]4)[cH:17][s:18]3)[cH:19][cH:20]2)cc1>>[OH:1][c:2]1[cH:3][cH:4][c:5]([NH:6][c:7]2[n:8][c:9](-[c:10]3[cH:11][cH:12][c:13]([Cl:14])[cH:15][cH:16]3)[cH:17][s:18]2)[cH:19][cH:20]1', 
            'F[c:14]1[cH:13][cH:12][c:11]([NH:10][c:9]2[s:8][cH:7][c:6](-[c:5]3[cH:4][cH:3][c:2]([Cl:1])[cH:27][cH:26]3)[n:25]2)[cH:24][cH:23]1.[OH:15][CH2:16][c:17]1[cH:18][cH:19][cH:20][cH:21][cH:22]1>>[Cl:1][c:2]1[cH:3][cH:4][c:5](-[c:6]2[cH:7][s:8][c:9]([NH:10][c:11]3[cH:12][cH:13][c:14]([O:15][CH2:16][c:17]4[cH:18][cH:19][cH:20][cH:21][cH:22]4)[cH:23][cH:24]3)[n:25]2)[cH:26][cH:27]1', 
            '[C:7](#[N:8])[S-:18].[F:1][c:2]1[cH:3][cH:4][c:5]([NH2:6])[cH:19][cH:20]1.O=[C:9]([c:10]1[cH:11][cH:12][c:13]([Cl:14])[cH:15][cH:16]1)[CH2:17]Br>>[F:1][c:2]1[cH:3][cH:4][c:5]([NH:6][c:7]2[n:8][c:9](-[c:10]3[cH:11][cH:12][c:13]([Cl:14])[cH:15][cH:16]3)[cH:17][s:18]2)[cH:19][cH:20]1'
            ]
        graph = RxnGraphEnumerator(route, mapper=self.mapper)
        real = Chem.MolFromSmiles("OCc1ccccc1")
        decoy = Chem.MolFromSmiles("OCc1cc(O)ccc1")
        
        #TODO: split into separate test case

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        catalog = FilterCatalog(params)

        graph.apply_pharma_filters(catalog)

        
        for l in graph.leaves:
            if l['smiles'] == "OCc1ccccc1":
                s = Chem.MolFromSmarts(l['query'])
                assert real.HasSubstructMatch(s)
                assert not decoy.HasSubstructMatch(s)
        
if __name__ == "__main__":
    pytest.main([__file__])
