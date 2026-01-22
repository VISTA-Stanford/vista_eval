from google.cloud import bigquery

def list_structure(project_id="som-nero-plevriti-deidbdf"):
    client = bigquery.Client(project=project_id)
    
    print(f"Scanning Project: {project_id}...\n")
    
    try:
        datasets = list(client.list_datasets())
        if not datasets:
            print("No datasets found! Check your permissions or project ID.")
            return

        for dataset in datasets:
            print(f"Dataset: {dataset.dataset_id}")
            print("-" * 30)
            
            # List tables inside this dataset
            tables = list(client.list_tables(dataset.dataset_id))
            if not tables:
                print("  (No tables found)")
            
            for table in tables:
                # This prints the EXACT reference you need to use
                print(f"  Table ID: {table.table_id}")
                print(f"  Full Ref: {project_id}.{dataset.dataset_id}.{table.table_id}")
            print("\n")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_structure()