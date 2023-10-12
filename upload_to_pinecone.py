
from dotenv import load_dotenv
from itertools import islice
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Pinecone
import os, time
import pandas as pd
import pinecone
from tqdm import tqdm

load_dotenv()

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name,
                         open_ai_api_key=os.getenv('OPENAI_API_KEY'))

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment="gcp-starter"
    )

db = Pinecone.from_existing_index(index_name='food-advisor', embedding=embed)

df = pd.read_csv('recipes_final.csv')
df = df.fillna('')

def tuple_to_document(t: tuple):
    
    return Document(
        page_content = t.Description,
        metadata = {
            'name':t.Name,
            'time':t.TotalTime,
            'category':t.RecipeCategory,
            'keywords':t.Keywords,
            'ingredients':t.Ingredients,
            'calories':t.Calories,
            'carbohydrates percentage':t.CarbohydratePercentage,
            'proteins percentage':t.ProteinPercentage,
            'fat percentage':t.FatPercentage,
            'sugar percentage':t.SugarPercentage,
            'instructions':t.RecipeInstructions,
            'yields':t.RecipeYield
        }
    )
    
docs = [tuple_to_document(row) for row in tqdm(df.itertuples(index=False))]

def chunks(lst, chk_size):
    lst_it = iter(lst)
    return iter(lambda: tuple(islice(lst_it, chk_size)), ())

chunkSize = 100
docs_chunked_list = list(chunks(docs, chunkSize))


print('Uploading to vector db')
s = time.perf_counter()
for docs in docs_chunked_list:
    db.add_documents(docs)
elapsed = time.perf_counter() - s
print("\033[1m" + f"Upload executed in {elapsed:0.2f} seconds." + "\033[0m")