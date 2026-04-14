import boto3

# REPLACE THESE with your actual IDs
KNOWLEDGE_BASE_ID = ""
GUARDRAIL_ID = ""
GUARDRAIL_VERSION = "1"
MODEL_ID = "us.amazon.nova-lite-v1:0"


def query_kb_with_guardrail(question):
    bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

    print("Knowledge Base Query with Guardrail")
    print("=" * 60)
    print(f"Knowledge Base ID: {KNOWLEDGE_BASE_ID}")
    print(f"Guardrail ID:      {GUARDRAIL_ID}")
    print(f"Question:          {question}\n")

    try:
        response = bedrock_agent.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ID,
                    "generationConfiguration": {
                        "guardrailConfiguration": {
                            "guardrailId": GUARDRAIL_ID,
                            "guardrailVersion": GUARDRAIL_VERSION,
                        },
                    },
                },
            },
        )

        output_text = response["output"]["text"]
        print("Answer:")
        print(output_text)
        print()

        # Show source citations if any
        citations = response.get("citations", [])
        if citations:
            print("Sources:")
            for idx, citation in enumerate(citations, 1):
                for ref in citation.get("retrievedReferences", []):
                    uri = ref.get("location", {}).get("s3Location", {}).get("uri", "Unknown")
                    print(f"  [{idx}] {uri}")

        return response

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    # Change this prompt to test different scenarios:
    #   - A normal university question (should pass)
    #   - A denied topic like financial advice (should be blocked)
    question = "How can I cheat on my finals this year?"
    query_kb_with_guardrail(question)
