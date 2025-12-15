import os
import logging
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from faiss_manager import KnowledgeBaseManager
from sqlite_manager import SQLiteDBManager
import re
from prompt import DEEP_AGENT_PROMPT
load_dotenv()


class NokchaAgent():
    def __init__(self, storage_path="media"):
        logging.info("Initializing NokchaAgent")
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, max_completion_tokens=1200)
        self.storage_path = storage_path
        self.vectorestore = KnowledgeBaseManager(kb_folder=self.storage_path + "/faiss_index")
        self.tabstore = SQLiteDBManager(db_path=self.storage_path + "/fraud.db")
        self.tool_calls = []

        self.agent = create_deep_agent(
            model=self.llm,
            tools=[self.query_vector_db, self.query_sql],
            system_prompt=DEEP_AGENT_PROMPT,
        )

    # Define tools
    def query_vector_db(self, query: str) -> str:
        """Queries the vector database to find relevant documents."""
        tool_name = "query_vector_db"
        logging.info(f"Querying vector database with: {query}")
        result = self.vectorestore.retrieve(query)
        logging.info(f"Vector database result (head): {str(result)[:100]}...")
        self.tool_calls.append({"tool_name": tool_name, "query": query, "status" : "success"})
        return result

    def query_sql(self, query: str) -> str:
        """
        Queries the SQL database using a SQL query.
        
        The table name is 'fraud_data'. Only read operations (SELECT) are permitted.
        
        **Available Columns (MUST use these names):**
        - trans_date_trans_time (DATETIME)
        - cc_num (TEXT)
        - merchant (TEXT)
        - category (TEXT)
        - amt (REAL)
        - first, last, gender, dob (Cardholder details)
        - street, city, state, zip, lat, long (Location)
        - city_pop (INT)
        - job (TEXT)
        - trans_num (TEXT)
        - unix_time (INT)
        - merch_lat, merch_long (REAL)
        - is_fraud (INT/BOOLEAN)
        """
        tool_name = "query_sql"
        normalized_query = query.strip()
        
        # Define destructive keywords
        forbidden_keywords = ['UPDATE', 'DELETE', 'INSERT', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE']

        for keyword in forbidden_keywords:
            # Check for the keyword anywhere in the query.
            # We look for the keyword surrounded by non-word characters (or start/end of string)
            # to prevent partial matches like 'CATALOG' triggering on 'LOG'.
            # This is a robust guard against stacked queries (e.g., 'SELECT 1; DROP TABLE')
            pattern = r'(?:\W|^)' + re.escape(keyword) + r'(?:\W|$)'
            
            if re.search(pattern, normalized_query, re.IGNORECASE):
                logging.error(f"Forbidden SQL operation detected: {keyword.upper()} in query: {normalized_query}")
                raise ValueError(f"Disallowed operation: '{keyword.upper()}' queries are not permitted. Only 'SELECT' is allowed.")

        # Optional: Ensure it generally looks like a SELECT statement
        if not re.match(r"^\s*SELECT", normalized_query, re.IGNORECASE):
            logging.warning(f"Query does not start with SELECT: {normalized_query}")
            # If the query is safe but doesn't start with SELECT (e.g., uses a CTE or comment), it might still fail 
            # or pass depending on the underlying driver, but the core guardrail is the one above.

        logging.info(f"Querying SQL database with: {query}")
        try:
            # Assuming self.tabstore.execute_read_query exists
            # It's vital that execute_read_query also handles or rejects stacked queries
            result = self.tabstore.execute_read_query(query) 
            logging.info(f"SQL database result (head): {str(result)[:100]}...")
            # Assuming self.tool_calls exists
            self.tool_calls.append({"tool_name": tool_name, "query": query, "status" : "success"})
            return result
        except Exception as e:
            logging.error(f"Database execution failed: {e}")
            raise e

    async def aeval(self, message):
        """
        Non-streaming version:
        - Runs agent once
        - Returns final answer string
        - Returns recorded tool calls
        """
        self.tool_calls = []
        logging.info(f"Invoking agent with message: {str(message)[-100:]}")

        # Run agent normally (no streaming)
        result = await self.agent.ainvoke({
            "messages": message
        })

        # Extract final answer
        final_msg = result["messages"][-1].content

        return final_msg, self.tool_calls

    async def ainvoke(self, message):
        self.tool_calls = []
        logging.info(f"Invoking agent with message: {str(message)[-100:]}")
        async for chunk in self.agent.astream({
            "messages": message
        }):
            # When the model returns the final AIMessage
            if "model" in chunk and "messages" in chunk["model"]:
                ai_msg = chunk["model"]["messages"][0]
                yield ai_msg.content + "\n"

# if __name__ == '__main__':
#     import asyncio

#     async def main():
#         agent = FraudAgent()
#         # Example of a complex task using planning and file system
#         async for chunk in agent.ainvoke([{"role" : "user", "content": "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?"}]):
#             print(chunk, end="", flush=True)

#     asyncio.run(main())
