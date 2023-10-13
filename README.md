# 🥘 Food Advisor

  This repo contains a streamlit application powered by LangChain.
  
  * A foods.com recipes dataset is contained in a vector database hosted in Pinecone.
  * The application returns the most suitable recipes according to the user's preferences.
  * The user inputs human-language text explaining what he wants for his meal.
  * The LLM interprets the user's taste and elaborate a string that will be used to query the vector database.
  * The similarity search yielded by the LLM query will return the most interesting recipes.

There's a snippet of the app below.
    
![App](https://github.com/matheusfmello/foodAdvisor/assets/71409156/68379dfc-a60a-4ea6-96d0-d7dd824c1f68)
