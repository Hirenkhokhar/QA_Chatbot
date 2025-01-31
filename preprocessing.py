import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = Client(Settings(persist_directory="db"))


numerical_columns_query = [
    'total_quantite Calculated', 'poids_brut_total_pesee', 'montant_fret',
    'poids_net_kg', 'montant_total_facture', 'poids_net', 'quantite'
]

numerical_columns_item_details = [
    'qte_facturee', 'montant_facture_par_ligne', 'montant_unitaire_par_ligne_facture'
]  

def load_data(file_path):
    return pd.ExcelFile(file_path)

def process_query_sheet(data, chroma_client):
    print("Processing 'query' sheet...")
    sheet_data = data.parse('query')

    sheet_data['text'] = sheet_data[
        ['exportateur', 'expediteur', 'destinataire', 'adresse_expediteur', 'origine',
         'adresse_destinataire', 'devise', 'paiement', 'importateur', 'client',
         'conditions_livraison', 'accords']
    ].astype(str).agg(lambda x: " | ".join(x), axis=1)

    collection = chroma_client.get_or_create_collection(name="export_query_data_with_numericals")

    for idx, row in sheet_data.iterrows():
        text = row['text']
        metadata = {
            "id": f"query_{idx}",
            "exportateur": row.get('exportateur', None),
            "destinataire": row.get('destinataire', None),
            **{col: row[col] for col in numerical_columns_query if col in row}
        }
        embedding = embedding_model.encode(text).tolist()
        collection.add(documents=[text], metadatas=[metadata], ids=[str(metadata['id'])])

    print("'query' sheet processed successfully!")

def process_item_details_sheet(data, chroma_client):
    print("Processing 'item_details' sheet...")
    sheet_data = data.parse('items_details')  
    sheet_data['text'] = sheet_data[
        ["Nom",'ref_client / Part Number', 'designation_facture']
    ].astype(str).agg(lambda x: " | ".join(x), axis=1)

    collection = chroma_client.get_or_create_collection(name="items_details_data")

    for idx, row in sheet_data.iterrows():
        text = row['text']
        metadata = {
            "id": f"item_details_{idx}",
            "sheet_name": "items_details",
            "ref_client": row.get('ref_client / Part Number', ""),
            "designation_facture": row.get('designation_facture', ""),
        }

        embedding = embedding_model.encode(text).tolist()
        collection.add(documents=[text], metadatas=[metadata], ids=[metadata['id']])

    print("'item_details' sheet processed and stored in ChromaDB!")

if __name__ == "__main__":
    file_path = "Espace_Original_Batch1.xlsx"
    data = load_data(file_path)
    process_query_sheet(data, chroma_client)
    process_item_details_sheet(data, chroma_client)
    print("All sheets processed and stored in ChromaDB!")
