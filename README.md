## zip-scrapings-huge

Repo for preprocessing and curation of large volumes of JPEGs. Contains code to curate a folder of JPEGs and create HDF5 datasets compatible with the DINO training in the [SCAMPI DINO](https://github.com/IverMartinsen/scampi-dino) repo.

### Move a huge folder of JPEGs into multiple subfolders (shards) to ease processing
```
move_jpgs_into_shards.py
```

### Create HDF5 datasets from multiple shards of JPEGs
```
create_hdf5_dataset.py
```
