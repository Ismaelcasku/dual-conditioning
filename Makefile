.PHONY: install test validate-config frozen-hashes audit-campaign \
        figures figure1 figure2 figure3 figure4 figure5 supp-compatibility \
        pymol-figure1-renderings pymol-figure3-renderings \
        pymol-figure5-renderings clean

PYTHON ?= python
PYMOL ?= pymol

install:
	$(PYTHON) -m pip install -e '.[guidance,figures,test]'

test:
	$(PYTHON) -m pytest

validate-config:
	$(PYTHON) scripts/analysis/validate_config.py configs/exp06_generation.yaml

frozen-hashes:
	cd reproducibility/exp06_frozen && sha256sum -c SHA256SUMS.txt

audit-campaign:
	@test -n "$(PROJECT_ROOT)" || \
	  (echo "Set PROJECT_ROOT=/path/to/dual_conditioning_archive"; exit 2)
	$(PYTHON) scripts/analysis/audit_campaign.py \
	  --manifest reproducibility/exp06_frozen/manifests/experimental_design_release_75.tsv \
	  --project-root "$(PROJECT_ROOT)" \
	  --output results/source_data/fragment_audit_per_molecule_v2.tsv \
	  --no-allow-protrude-reordering

figures: figure1 figure2 figure3 figure4 supp-compatibility

figure1:
	$(PYTHON) scripts/figures/figure1_protocol.py

figure2:
	$(PYTHON) scripts/figures/figure2_generation_outcomes.py

figure3:
	$(PYTHON) scripts/figures/figure3_fragmentation.py

figure4:
	$(PYTHON) scripts/figures/figure4_relaxation_outcome_map.py

figure5:
	$(PYTHON) scripts/figures/figure5_structural_examples.py

supp-compatibility:
	$(PYTHON) scripts/figures/supp_pair_compatibility.py

pymol-figure1-renderings:
	$(PYMOL) -cq reproducibility/pymol/render_figure1_structures.pml

pymol-figure3-renderings:
	$(PYMOL) -cq reproducibility/pymol/render_figure3_fragmentation_example.pml

pymol-figure5-renderings:
	$(PYMOL) -cq reproducibility/pymol/render_figure5_structural_examples.pml

clean:
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
