# Commands for GCP

## sync from gs bucket

gsutil -m rsync -r gs://<bucket_name>/ <path_on_VM>

ex) For vista_bench bucket

gsutil -m rsync -r gs://vista_bench/ vista_bench

## Unstage last commit

git reset HEAD~1