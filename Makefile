red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@


# Extraction and slicing for Segthor

## Full training set
data/segthor_train_full: data/segthor_train_full.zip
	@echo "$(yellow)unzip $<$(reset)"
	unzip -q $< -d $@
	find $@ -name .DS_Store -delete

# Shared settings keep patient splits identical across all 12 combinations.
PYTHON ?= python
SEGTHOR_SOURCE ?= data/segthor_train_full
RETAINS ?= 5
SEED ?= 0
FOLD ?= 0
PROCESS ?= 1

# Full slices, no resampling (existing output paths).
data/SEGTHOR: PREPROCESS_FLAGS = --resample none
data/SEGTHOR_hu: PREPROCESS_FLAGS = --resample none --hu_clip

# Full slices, XY resampling; second variant also clips HU.
data/full_xy/SEGTHOR: PREPROCESS_FLAGS = --resample xy
data/full_hu_xy/SEGTHOR: PREPROCESS_FLAGS = --resample xy --hu_clip

# Full slices, XYZ resampling; second variant also clips HU.
data/full_xyz/SEGTHOR: PREPROCESS_FLAGS = --resample xyz
data/full_hu_xyz/SEGTHOR: PREPROCESS_FLAGS = --resample xyz --hu_clip

# Tiles, no resampling; second variant also clips HU.
data/tiled/SEGTHOR: PREPROCESS_FLAGS = --resample none --tiling
data/tiled_hu/SEGTHOR: PREPROCESS_FLAGS = --resample none --tiling --hu_clip

# Tiles, XY resampling; second variant also clips HU.
data/tiled_xy/SEGTHOR: PREPROCESS_FLAGS = --resample xy --tiling
data/tiled_hu_xy/SEGTHOR: PREPROCESS_FLAGS = --resample xy --tiling --hu_clip

# Tiles, XYZ resampling; second variant also clips HU.
data/tiled_xyz/SEGTHOR: PREPROCESS_FLAGS = --resample xyz --tiling
data/tiled_hu_xyz/SEGTHOR: PREPROCESS_FLAGS = --resample xyz --tiling --hu_clip

SEGTHOR_DATASETS := data/SEGTHOR data/SEGTHOR_hu \
    data/full_xy/SEGTHOR data/full_hu_xy/SEGTHOR \
    data/full_xyz/SEGTHOR data/full_hu_xyz/SEGTHOR \
    data/tiled/SEGTHOR data/tiled_hu/SEGTHOR \
    data/tiled_xy/SEGTHOR data/tiled_hu_xy/SEGTHOR \
    data/tiled_xyz/SEGTHOR data/tiled_hu_xyz/SEGTHOR

# Example: make data/tiled_hu_xyz/SEGTHOR
# Uses 256x256 slices or 256x256 tiles with stride 128x128 (script defaults).
# XY/XYZ target spacing and HU bounds are derived from training patients only.
# Existing datasets are reused; settings changes require a fresh output directory.
# A leftover _tmp directory is preserved for inspection after a failed run.
$(SEGTHOR_DATASETS): $(SEGTHOR_SOURCE)
	@test ! -e "$@" || { echo "Output already exists: $@"; exit 1; }
	@test ! -e "$@_tmp" || { echo "Temporary output already exists: $@_tmp"; exit 1; }
	mkdir -p "$(@D)"
	$(PYTHON) $(CFLAGS) slice_segthor.py --source_dir "$<" --dest_dir "$@_tmp" \
		--retains $(RETAINS) --seed $(SEED) --fold $(FOLD) --process $(PROCESS) --save_metadata \
		$(PREPROCESS_FLAGS)
	mv "$@_tmp" "$@"

# Optional: prepare every combination (can require substantial time and storage).
.PHONY: preprocess-all
preprocess-all: $(SEGTHOR_DATASETS)

# Train nested outputs with --dataset SEGTHOR and their parent as --data_root.
# Example: python main.py --dataset SEGTHOR --data_root data/tiled_hu_xyz \
#   --dest results/tiled_hu_xyz --tiling --loss ce --gpu
# Legacy data/SEGTHOR_hu uses --dataset SEGTHOR_hu --data_root data.
