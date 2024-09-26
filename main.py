from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from vanna.groq import Groq

class MyVanna(ChromaDB_VectorStore, Groq):
    def __init__(self, config=None):
        if config is None:
            config = {}

        ChromaDB_VectorStore.__init__(self, config={"path": "./chroma_models/metals_model2"})
        api_key=config.get("api_key","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")
        model=config.get("model","llama-3.1-70b-versatile")

        Groq.__init__(self,
                      config={"api_key":api_key,"model":model,"temperature":0.1,"top_k":50}
                      
                     )
        #Groq.__init__(self, config=config)
 
vn = MyVanna()

'''

import csv
 
# Replace 'your_file_path.csv' with the actual path to your CSV file
file_path = 'training_data.csv'
 
# Open the CSV file and read its contents
with open(file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data_type = row['training_data_type']
        question = row['question']
        content = row['content']
       
        if data_type.lower() == 'ddl':
            vn.train(ddl=content)
        elif data_type.lower() == 'documentation':
            vn.train(documentation=content)
        elif data_type.lower() == 'sql':
            vn.train(question=question, sql=content)
        else:
            print(f"Unknown data type: {data_type}")


'''
print(vn.get_related_documentation(question="What is the current status of production, payment, and delivery for all promised orders?"))

print(vn.generate_sql(question="What is the current status of production, payment, and delivery for all promised orders?"))