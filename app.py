import datetime
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
import langchain
import pinecone
import os
import streamlit as st

from chain import build_query_chain

load_dotenv()

def fix_prep_time(obj):
    if isinstance(obj, datetime.datetime):
        return f"{obj.hour}:{obj.minute}"
    else:
        return obj


llm = OpenAI(temperature=0.0)

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name,
                         open_ai_api_key=os.getenv('OPENAI_API_KEY'))
vectordb = Pinecone.from_existing_index(index_name="food-advisor", embedding=embed)


def main():
    st.title("Food Advisor")

    st.write("Fill in the form in the sidebar to get a recipe recommendation")

    # Ask the questions and store answers
    st.sidebar.subheader("1. Meal")
    meal = st.sidebar.selectbox(
        "Kind of meal", ["Select an option", "Breakfast", "Lunch", "Dinner", "Dessert", "Snack", "Drinks"]
    )

    st.sidebar.subheader("2. Maximum Preparation Time")
    time = st.sidebar.time_input(
        "Time you are willing to spend",
        datetime.time(0,0)
    )

    st.sidebar.subheader("3. Ingredients to Include")
    included_ingredients = st.sidebar.text_input(
        "Include those ingredients",
        value=""
    )

    st.sidebar.subheader("4. Ingredients to Exclude")
    excluded_ingredients = st.sidebar.text_input(
        "Exclude those ingredients",
        value=""
    )

    st.sidebar.subheader("5. Describe what kind of food you are into")
    description = st.sidebar.text_input("Complement the answers above")

    query_chain, query_response_format, query_output_parser = build_query_chain(llm)

    if st.button("Generate recommendation"):
        response = query_chain.run(
            {
                'food_category':meal,
                'preparation_time':time,
                'included_ingredients':included_ingredients,
                'excluded_ingredients':excluded_ingredients,
                'description':description,
                "response_format": query_response_format
            }
        )

        query = query_output_parser.parse(response)["query_string"]
        print(query)
        docs = vectordb.similarity_search(query=query, k=5)

        recipe_options = [
            {
                "name": doc.metadata["name"],
                "time": fix_prep_time(doc.metadata['time']),
                #"text": doc.metadata["text"],
                "carbohydrates": doc.metadata["carbohydrates percentage"],
                "protein": doc.metadata["proteins percentage"],
                "fat": doc.metadata["fat percentage"],
                "sugar": doc.metadata["sugar percentage"],
                "instructions": doc.metadata["instructions"].replace('\n', '').split('.')[:-1],
                "ingredients": eval(doc.metadata["ingredients"])
            }
            for doc in docs
        ]

        recipe_1 = recipe_options[0]
        recipe_2 = recipe_options[1]
        recipe_3 = recipe_options[2]
        recipe_4 = recipe_options[3]
        recipe_5 = recipe_options[4]
        
        st.write("---")
        
        st.write("### 🥘 **Recommended Recipes**")
        
        
        
        try:
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([f"Option {i+1}" for i in range(5)])
            
            with tab1:
                
                st.write(
                    f"""
                        **{recipe_1['name']}**
                        
                        {recipe_1['time']}
                        
                        """
                )
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Carbohydrates", f"{round(100*recipe_1['carbohydrates'], 2)}%")
                col2.metric("Protein", f"{round(100*recipe_1['protein'])}%")
                col3.metric("Fat", f"{round(100*recipe_1['fat'])}%")
                col4.metric("Sugar", f"{round(100*recipe_1['sugar'])}%")
                
                ingredientsCol, instructionsCol = st.columns(2)
                
                formatIngredients = [f"* {key}: {value}\n" for key, value in recipe_1['ingredients'].items()]
                formatInstructions = [f"1. {step}.\n" for step in recipe_1['instructions']]
                
                ingredientsCol.write(
                    "".join(formatIngredients)
                )
                
                instructionsCol.write(
                    "".join(formatInstructions)
                )
                
            with tab2:
                
                st.write(
                    f"""
                        **{recipe_2['name']}**
                        
                        {recipe_2['time']}
                        
                        """
                )
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Carbohydrates", f"{round(100*recipe_2['carbohydrates'], 2)}%")
                col2.metric("Protein", f"{round(100*recipe_2['protein'])}%")
                col3.metric("Fat", f"{round(100*recipe_2['fat'])}%")
                col4.metric("Sugar", f"{round(100*recipe_2['sugar'])}%")
                
                ingredientsCol, instructionsCol = st.columns(2)
                
                formatIngredients = [f"* {key}: {value}\n" for key, value in recipe_2['ingredients'].items()]
                formatInstructions = [f"1. {step}\n" for step in recipe_2['instructions']]
                
                ingredientsCol.write(
                    "".join(formatIngredients)
                )
                
                instructionsCol.write(
                    "".join(formatInstructions)
                )
                
            with tab3:
                
                st.write(
                    f"""
                        **{recipe_3['name']}**
                        
                        {recipe_3['time']}
                        
                        """
                )
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Carbohydrates", f"{round(100*recipe_3['carbohydrates'], 2)}%")
                col2.metric("Protein", f"{round(100*recipe_3['protein'])}%")
                col3.metric("Fat", f"{round(100*recipe_3['fat'])}%")
                col4.metric("Sugar", f"{round(100*recipe_3['sugar'])}%")
                
                ingredientsCol, instructionsCol = st.columns(2)
                
                formatIngredients = [f"* {key}: {value}\n" for key, value in recipe_3['ingredients'].items()]
                formatInstructions = [f"1. {step}\n" for step in recipe_3['instructions']]
                
                ingredientsCol.write(
                    "".join(formatIngredients)
                )
                
                instructionsCol.write(
                    "".join(formatInstructions)
                )
            
            with tab4:
                
                st.write(
                    f"""
                        **{recipe_4['name']}**
                        
                        {recipe_4['time']}
                        
                        """
                )
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Carbohydrates", f"{round(100*recipe_4['carbohydrates'], 2)}%")
                col2.metric("Protein", f"{round(100*recipe_4['protein'])}%")
                col3.metric("Fat", f"{round(100*recipe_4['fat'])}%")
                col4.metric("Sugar", f"{round(100*recipe_4['sugar'])}%")
                
                ingredientsCol, instructionsCol = st.columns(2)
                
                formatIngredients = [f"* {key}: {value}\n" for key, value in recipe_4['ingredients'].items()]
                formatInstructions = [f"1. {step}\n" for step in recipe_4['instructions']]
                
                ingredientsCol.write(
                    "".join(formatIngredients)
                )
                
                instructionsCol.write(
                    "".join(formatInstructions)
                )
                
            with tab5:
                
                st.write(
                    f"""
                        **{recipe_5['name']}**
                        
                        {recipe_5['time']}
                        
                        """
                )
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Carbohydrates", f"{round(100*recipe_5['carbohydrates'], 2)}%")
                col2.metric("Protein", f"{round(100*recipe_5['protein'])}%")
                col3.metric("Fat", f"{round(100*recipe_5['fat'])}%")
                col4.metric("Sugar", f"{round(100*recipe_5['sugar'])}%")
                
                ingredientsCol, instructionsCol = st.columns(2)
                
                formatIngredients = [f"* {key}: {value}\n" for key, value in recipe_5['ingredients'].items()]
                formatInstructions = [f"1. {step}\n" for step in recipe_5['instructions']]
                
                ingredientsCol.write(
                    "".join(formatIngredients)
                )
                
                instructionsCol.write(
                    "".join(formatInstructions)
                )
            
                
        except:
            st.write(
                f"""
                    
                    """
            )


if __name__ == "__main__":
    main()