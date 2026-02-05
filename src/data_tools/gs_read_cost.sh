#!/bin/bash

# Define your source paths
PATHS=(
  "gs://vista_bench"
  "gs://su-vista-uscentral1/chaudhari_lab/ct_data/ct_scans/vista/nov25"
)

total_size_gb=0

echo "---------------------------------------------------"
echo "Calculating sizes and checking locations..."
echo "---------------------------------------------------"

for path in "${PATHS[@]}"; do
  # 1. Get the bucket name from the path for location check
  bucket=$(echo "$path" | cut -d'/' -f3)

  # 2. Get the location of the bucket
  location=$(gsutil ls -L -b "gs://$bucket" | grep "Location constraint:" | awk '{print $3}')

  # 3. Get the size of the data in bytes
  # 'du -s' gives summary, tail -1 grabs the total line, cut extracts the byte count
  bytes=$(gsutil du -s "$path" | tail -1 | awk '{print $1}')
  
  # Convert to GB (1 GB = 10^9 bytes for billing, though usually GiB 2^30 is used in tools. 
  # GCP Billing often uses Gibibytes (GiB). We will calculate in GiB for accuracy.)
  gb=$(echo "scale=4; $bytes / 1073741824" | bc)
  
  # Add to total
  total_size_gb=$(echo "scale=4; $total_size_gb + $gb" | bc)

  echo "Path:     $path"
  echo "Location: $location"
  echo "Size:     $gb GiB"
  echo "---------------------------------------------------"
done

echo "TOTAL DATA TO READ: $total_size_gb GiB"
echo "---------------------------------------------------"