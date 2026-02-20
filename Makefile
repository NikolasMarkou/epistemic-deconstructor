# Makefile for Epistemic Deconstructor Claude Skill
# Packages the repository into a distributable Claude skill format

SKILL_NAME := epistemic-deconstructor
VERSION := 6.8.0
BUILD_DIR := build
DIST_DIR := dist

# Files to include in the skill package
SKILL_FILE := src/SKILL.md
REFERENCE_FILES := $(wildcard src/references/*.md)
SCRIPT_FILES := $(wildcard src/scripts/*.py)
DOC_FILES := README.md LICENSE CHANGELOG.md

# Default target
.PHONY: all
all: package

# Build the skill package structure
.PHONY: build
build:
	@echo "Building skill package: $(SKILL_NAME)"
	mkdir -p $(BUILD_DIR)/$(SKILL_NAME)
	mkdir -p $(BUILD_DIR)/$(SKILL_NAME)/references
	mkdir -p $(BUILD_DIR)/$(SKILL_NAME)/scripts
	mkdir -p $(BUILD_DIR)/$(SKILL_NAME)/config
	@# Copy main skill file
	cp $(SKILL_FILE) $(BUILD_DIR)/$(SKILL_NAME)/
	@# Copy reference files
	cp $(REFERENCE_FILES) $(BUILD_DIR)/$(SKILL_NAME)/references/
	@# Copy scripts
	cp $(SCRIPT_FILES) $(BUILD_DIR)/$(SKILL_NAME)/scripts/
	@# Copy config
	cp src/config/domains.json $(BUILD_DIR)/$(SKILL_NAME)/config/
	@# Copy documentation
	cp $(DOC_FILES) $(BUILD_DIR)/$(SKILL_NAME)/ 2>/dev/null || true
	@echo "Build complete: $(BUILD_DIR)/$(SKILL_NAME)"

# Create a combined single-file skill (SKILL.md with references inlined)
.PHONY: build-combined
build-combined:
	@echo "Building combined single-file skill..."
	mkdir -p $(BUILD_DIR)
	cp $(SKILL_FILE) $(BUILD_DIR)/$(SKILL_NAME)-combined.md
	@echo "" >> $(BUILD_DIR)/$(SKILL_NAME)-combined.md
	@echo "---" >> $(BUILD_DIR)/$(SKILL_NAME)-combined.md
	@echo "" >> $(BUILD_DIR)/$(SKILL_NAME)-combined.md
	@echo "# Bundled References" >> $(BUILD_DIR)/$(SKILL_NAME)-combined.md
	@for ref in $(REFERENCE_FILES); do \
		echo "" >> $(BUILD_DIR)/$(SKILL_NAME)-combined.md; \
		echo "---" >> $(BUILD_DIR)/$(SKILL_NAME)-combined.md; \
		echo "" >> $(BUILD_DIR)/$(SKILL_NAME)-combined.md; \
		cat $$ref >> $(BUILD_DIR)/$(SKILL_NAME)-combined.md; \
	done
	@echo "Combined skill created: $(BUILD_DIR)/$(SKILL_NAME)-combined.md"

# Package as zip for distribution
.PHONY: package
package: build
	@echo "Packaging skill as zip..."
	mkdir -p $(DIST_DIR)
	cd $(BUILD_DIR) && zip -r ../$(DIST_DIR)/$(SKILL_NAME)-v$(VERSION).zip $(SKILL_NAME)
	@echo "Package created: $(DIST_DIR)/$(SKILL_NAME)-v$(VERSION).zip"

# Package combined single-file version
.PHONY: package-combined
package-combined: build-combined
	mkdir -p $(DIST_DIR)
	cp $(BUILD_DIR)/$(SKILL_NAME)-combined.md $(DIST_DIR)/
	@echo "Combined skill copied to: $(DIST_DIR)/$(SKILL_NAME)-combined.md"

# Create tarball
.PHONY: package-tar
package-tar: build
	@echo "Packaging skill as tarball..."
	mkdir -p $(DIST_DIR)
	cd $(BUILD_DIR) && tar -czvf ../$(DIST_DIR)/$(SKILL_NAME)-v$(VERSION).tar.gz $(SKILL_NAME)
	@echo "Package created: $(DIST_DIR)/$(SKILL_NAME)-v$(VERSION).tar.gz"

# Validate skill structure
.PHONY: validate
validate:
	@echo "Validating skill structure..."
	@test -f $(SKILL_FILE) || (echo "ERROR: $(SKILL_FILE) not found" && exit 1)
	@grep -q "^name:" $(SKILL_FILE) || (echo "ERROR: SKILL.md missing 'name' in frontmatter" && exit 1)
	@grep -q "^description:" $(SKILL_FILE) || (echo "ERROR: SKILL.md missing 'description' in frontmatter" && exit 1)
	@test -d src/references || (echo "ERROR: src/references/ directory not found" && exit 1)
	@test -d src/scripts || (echo "ERROR: src/scripts/ directory not found" && exit 1)
	@echo "Validation passed!"

# Check Python syntax in scripts
.PHONY: lint
lint:
	@echo "Checking Python syntax..."
	@for script in $(SCRIPT_FILES); do \
		echo "  py_compile $$script"; \
		python3 -m py_compile $$script || exit 1; \
	done
	@echo "Syntax check passed!"

# Run tests
.PHONY: test
test: lint
	@echo "Running unit tests..."
	python3 -m unittest discover -s tests -v
	@echo "Running --help smoke tests..."
	@for script in $(SCRIPT_FILES); do \
		echo "  $$script --help"; \
		python3 $$script --help > /dev/null || exit 1; \
	done
	@echo "Tests passed!"

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(DIST_DIR)
	rm -f hypotheses.json
	@echo "Clean complete"

# Show package contents
.PHONY: list
list: build
	@echo "Package contents:"
	@find $(BUILD_DIR)/$(SKILL_NAME) -type f | sort

# Install skill to Claude config (placeholder - adjust path as needed)
.PHONY: install
install: build
	@echo "Installing skill..."
	@echo "Copy $(BUILD_DIR)/$(SKILL_NAME) to your Claude skills directory"
	@echo "Or use: make package && unzip dist/$(SKILL_NAME)-v$(VERSION).zip -d ~/.claude/skills/"

# Sync skill to Claude config directory
SKILL_DEST := $(HOME)/.claude/skills/$(SKILL_NAME)

.PHONY: sync-skill
sync-skill:
	@echo "Syncing skill to $(SKILL_DEST)..."
	mkdir -p $(SKILL_DEST)/references $(SKILL_DEST)/scripts $(SKILL_DEST)/config
	cp src/SKILL.md $(SKILL_DEST)/
	cp src/references/*.md $(SKILL_DEST)/references/
	cp src/scripts/*.py $(SKILL_DEST)/scripts/
	cp src/config/domains.json $(SKILL_DEST)/config/
	@echo "Skill synced to $(SKILL_DEST)"

# Help
.PHONY: help
help:
	@echo "Epistemic Deconstructor Skill - Makefile targets:"
	@echo ""
	@echo "  make build           - Build skill package structure"
	@echo "  make build-combined  - Build single-file skill with inlined references"
	@echo "  make package         - Create zip package (default)"
	@echo "  make package-combined - Create single-file skill package"
	@echo "  make package-tar     - Create tarball package"
	@echo "  make validate        - Validate skill structure"
	@echo "  make lint            - Check Python syntax"
	@echo "  make test            - Run tests"
	@echo "  make clean           - Remove build artifacts"
	@echo "  make list            - Show package contents"
	@echo "  make install         - Show install instructions"
	@echo "  make sync-skill      - Sync skill to ~/.claude/skills/"
	@echo "  make help            - Show this help"
	@echo ""
	@echo "Skill: $(SKILL_NAME) v$(VERSION)"
