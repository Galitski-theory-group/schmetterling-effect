.ONESHELL: # Applies to every target in the file!

PYTHON_VERSION ?= $(shell compgen -c python | sort -V | uniq | grep -E '^python[0-9]+\.[0-9]+$$' | tail -n 1 | cut -c7-)

# name
.buttefly:
	@echo "PYTHON_VERSION: $(PYTHON_VERSION)"
	python$(PYTHON_VERSION) -m venv .buttefly
	. .buttefly/bin/activate; .buttefly/bin/pip$(PYTHON_VERSION) install --upgrade pip$(PYTHON_VERSION) ; .buttefly/bin/pip$(PYTHON_VERSION) install -e .[dev,test]

brick: .buttefly

clean: .buttefly
	rm -rf .buttefly
