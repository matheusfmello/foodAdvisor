from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain, SequentialChain

def build_query_chain(llm):

    query_string = ResponseSchema(
        name="query_string",
        description="The string used to query the vector database in order to find the most suitable recipes"
    )

    output_parser = StructuredOutputParser.from_response_schemas(
        [query_string]
    )

    response_format = output_parser.get_format_instructions()

    prompt = PromptTemplate.from_template(
        template="""
        You are a food advisor. Your task is to pick from a recipes database the recipe that better fits the user preference.
        
        Take your time to understand the following user preferences:
        'food category':{food_category}
        'maximum preparation time':{preparation_time}
        'necessary ingredients':{included_ingredients}
        'excluded ingredients':{excluded_ingredients}
        'description':{description}
        
        Now take your time to gather those preferences and create a string to perform a similarity search on a vector database containing food recipes.
        The query must be clear and specific, utilizing relevant features.
        
        {response_format}
        """
    )

    query_chain = LLMChain(llm=llm, prompt=prompt, output_key='query')

    chain = SequentialChain(
        chains=[query_chain],
        input_variables=['food_category', 'preparation_time', 'included_ingredients', 'excluded_ingredients', 'description'] + [ 'response_format'],
        output_variables=['query'],
        verbose=False
    )
    
    return chain, response_format, output_parser
