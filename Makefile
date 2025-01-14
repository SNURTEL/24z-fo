DIR="data/IllustrisTNG"

.PHONY: data

data:
	mkdir -p $(DIR) && \
	wget --directory-prefix $(DIR) --input-file data_links.txt && \
	echo "done";