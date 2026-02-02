#!/usr/bin/env python3
"""
Test H2OGPTE agent with custom tool integration.
"""

import os
import time
from dotenv import load_dotenv
from h2ogpte import H2OGPTE

load_dotenv()

API_KEY = os.getenv('H2OGPTE_API_KEY')
ADDRESS = os.getenv('H2OGPTE_ADDRESS')

if not API_KEY or not ADDRESS:
    print("Error: H2OGPTE_API_KEY and H2OGPTE_ADDRESS must be set in .env file")
    exit(1)

print(f"Connecting to {ADDRESS}...")
client = H2OGPTE(address=ADDRESS, api_key=API_KEY)
print("Client created successfully")

# Create a collection for documents
collection_id = client.create_collection(
    name='SumOmniEval Agent',
    description='H2OGPTe Agent Tool Call Using SumOmniEval',
)
print(f"Collection created: {collection_id}")

# Upload and ingest file for agent use
with open('example_tool.py', 'rb') as f:
    upload_id = client.upload('example_tool.py', f)

ingest_job = client.ingest_uploads(
    collection_id=collection_id,
    upload_ids=[upload_id],
    ingest_mode='agent_only'
)

# Wait for ingestion to complete
print("Waiting for tool ingestion...")
while True:
    job_status = client.get_job(ingest_job.id)
    if job_status.completed:
        print("Ingestion complete")
        break
    if job_status.failed:
        print(f"Ingestion failed: {job_status.errors}")
        exit(1)
    time.sleep(2)

# Register the custom tool
client.add_custom_agent_tool(
    tool_type='general_code',
    tool_args={'tool_name': 'example_tool'},
    custom_tool_path='example_tool.py'
)

# Create chat session
chat_session_id = client.create_chat_session()
print(f"Chat session created: {chat_session_id}")

generated_summary = """
A massive fire broke out at a high-rise residential building in a major city during the early hours of the morning. The blaze spread with incredible speed, moving from the lower floors to the top of the 24-storey structure in a very short amount of time. Emergency services were called to the scene shortly after midnight and spent many hours trying to control the flames, while many residents were forced to find their own ways out of the building.
The incident resulted in numerous casualties and hospitalizations, with authorities confirming that several people had died and many more were in critical condition. Hundreds of residents were displaced by the fire, losing their homes and belongings. Community centers in the surrounding area were opened to provide temporary shelter and support for those who managed to escape.
Questions have been raised about the safety of the building and the materials used during a recent renovation. There are reports that residents had previously voiced concerns about fire risks, and many are now calling for a full investigation into how the fire was able to spread so quickly. While the exact cause remains under investigation, the event has highlighted significant issues regarding fire safety procedures in residential blocks.
"""
reference_summary = """
A massive fire broke out shortly after midnight at the 24-storey Grenfell Tower in North Kensington, West London, resulting in at least six confirmed deaths and 78 hospitalizations. Emergency services, including over 250 firefighters and 100 medics, arrived at the scene around 00:54 BST to find a "horror movie" scene. Residents were forced to make life-or-death decisions, with some ignoring official "stay put" advice to escape through smoke-filled corridors while others signaled for help from windows using torches and mobile phones.
The disaster has left hundreds of people homeless and many families desperately searching for missing relatives. Local community centers, churches, and volunteers have mobilized to provide food, shelter, and supplies to survivors who lost everything in the blaze. While a structural engineer has confirmed the building is not in immediate danger of collapse, the Metropolitan Police warn that the recovery operation will be lengthy and the death toll is expected to rise as they gain access to the upper floors.
In the wake of the tragedy, serious questions have emerged regarding the building's safety and a recent £10m refurbishment completed in 2016. Survivors reported that fire alarms failed to sound and that the fire spread with "extraordinary ferocity" across the building's exterior. The Grenfell Action Group had previously warned of fire risks and restricted emergency access, leading to calls from the London Mayor and local MPs for a full public inquiry into how such a catastrophic failure could occur.
"""

system_prompt = """
You have a tool called example_tool.py. When asked to process something, use this tool.
Usage: python example_tool.py 'input_string'
Always provide the output of the tool to the user.
"""

user_prompt = f"""
Given a generated summary and reference summary, call the relevant tools to calculate the ROUGE score.

Generated Summary:
{generated_summary}

Reference Summary:
{reference_summary}
"""

# Run agent query
print("Running agent query...")
with client.connect(chat_session_id) as session:
    reply = session.query(
        user_prompt,
        llm_args=dict(
            use_agent=True,
            agent_type="auto",
            agent_code_writer_system_message=system_prompt
        )
    )

print("\nAgent Response:")
print(reply.content)