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

data/SEGTHOR: data/segthor_train_full
	@echo "$(green)python $(CFLAGS) slice_segthor.py$(reset)"
	find $< -name .DS_Store -delete
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir $< --dest_dir $@_tmp \
		--shape 256 256 --retain 5
	mv $@_tmp $@

data/SEGTHOR_hu: data/segthor_train_full
	@echo "$(green)python $(CFLAGS) slice_segthor.py --hu_clip$(reset)"
	find $< -name .DS_Store -delete
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir $< --dest_dir $@_tmp \
		--shape 256 256 --retain 5 --hu_clip
	mv $@_tmp $@