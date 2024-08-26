# Data Engineering
## Introduction
**Why did the data engineer break up with the SQL query?

Because it couldn't join the right table!**

Welcome to the data engineering part of the project. Here, you'll find three folders corresponding to the Medallion Architecture:

Bronze: Raw files
Silver: Filtered and basic augmentation on files
Gold: Reporting-ready tables
Each folder contains subfolders for testing and workflow. The Testing folder showcases the data pipeline creation process and how we ingested the historical data. The Workflow folder contains the production-ready notebooks that we orchestrated via Databricks Workflows.

## Data Architecture
We chose a lakehouse approach using Databricks due to its native integration with tools like Delta Tables, Unity Catalog, and Databricks Workflows. These tools provided everything needed to build a fully functional lakehouse.

## Data Extraction
We primarily used Apache Spark to handle all of our transformations and extractions. For both the GDELT events and GKG tables ingestion process, we performed the following steps:

1. Consulted a table_control table, which returns the last update date for the table of interest.
2. Visited the URL, extracted the ZIP file, unzipped it, and deposited it in an S3 bucket.
3. Retrieved the Parquet file from the S3 bucket, enforced the Delta Live Table schema, and upserted the records based on unique IDs or attributes from the table.
4. Updated the table_control table with the corresponding date for the table that was inserted.

## Data Transformation
Occurring at the silver and gold layers, this workflow is automatically triggered after the successful execution of the insertion in the bronze layers. Delta Live Tables were also incorporated to allow for data versioning and data quality enforcement. Email notifications were sent in case of workflow success or failure. 

## Data Governance
Unity Catalog is our go-to solution for this approach. Although we didn't need to limit access to data, control groups can easily be set up with the Unity Catalog configuration we implemented.

## Next steps
- Complete workflow implementation for silver and gold tables: Finalize the end-to-end workflow to ensure smooth processing from the silver to gold layers.
- Enhance the date table for better catchup processes: Implement a more robust date table to handle catchup processes for any missing date values effectively.
- Separate models from notebook environments: Deploy models in dedicated Machine Learning modules within Databricks, separating them from typical notebook environments to streamline management and execution.
