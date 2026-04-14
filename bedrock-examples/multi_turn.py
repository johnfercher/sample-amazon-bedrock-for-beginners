import boto3

def multi_turn_conversation():
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    model_id = "us.amazon.nova-lite-v1:0"

    # System prompt sets the assistant's behavior
    system_prompt = [
        {
            "text": "You are a helpful cooking assistant. Provide concise recipe suggestions."
        }
    ]

    # Conversation history - we'll build this up with each turn
    conversation_history = []

    print("Multi-turn Conversation Demo")
    print("=" * 60)

    # Turn 1: Ask for recipe suggestions
    print("\nTurn 1: User asks for recipe suggestions")
    print("-" * 60)

    user_message_1 = "Suggest a quick dinner recipe with chicken."
    print(f"User: {user_message_1}")

    conversation_history.append({
        "role": "user",
        "content": [{"text": user_message_1}]
    })

    try:
        response_1 = bedrock_runtime.converse(
            modelId=model_id,
            system=system_prompt,
            messages=conversation_history,
            inferenceConfig={"temperature": 0.7, "maxTokens": 200}
        )

        assistant_message_1 = response_1['output']['message']['content'][0]['text']
        print(f"Assistant: {assistant_message_1}")

        # Add assistant's response to history
        conversation_history.append({
            "role": "assistant",
            "content": [{"text": assistant_message_1}]
        })

    except Exception as e:
        print(f"Error: {e}")
        return

    # Turn 2: Ask for modifications
    print("\n\nTurn 2: User asks for modifications")
    print("-" * 60)

    user_message_2 = "Can you make it vegetarian instead?"
    print(f"User: {user_message_2}")

    conversation_history.append({
        "role": "user",
        "content": [{"text": user_message_2}]
    })

    try:
        response_2 = bedrock_runtime.converse(
            modelId=model_id,
            system=system_prompt,
            messages=conversation_history,
            inferenceConfig={"temperature": 0.7, "maxTokens": 200}
        )

        assistant_message_2 = response_2['output']['message']['content'][0]['text']
        print(f"Assistant: {assistant_message_2}")

        # Add assistant's response to history
        conversation_history.append({
            "role": "assistant",
            "content": [{"text": assistant_message_2}]
        })

    except Exception as e:
        print(f"Error: {e}")
        return

    # Turn 3: Ask for cooking time
    print("\n\nTurn 3: User asks for cooking time")
    print("-" * 60)

    user_message_3 = "How long will this take to prepare?"
    print(f"User: {user_message_3}")

    conversation_history.append({
        "role": "user",
        "content": [{"text": user_message_3}]
    })

    try:
        response_3 = bedrock_runtime.converse(
            modelId=model_id,
            system=system_prompt,
            messages=conversation_history,
            inferenceConfig={"temperature": 0.7, "maxTokens": 200}
        )

        assistant_message_3 = response_3['output']['message']['content'][0]['text']
        print(f"Assistant: {assistant_message_3}")

        # Display token usage for final turn
        usage = response_3.get('usage', {})
        print(f"\nFinal Turn Token Usage:")
        print(f"  Input tokens: {usage.get('inputTokens', 'N/A')} (includes full history)")
        print(f"  Output tokens: {usage.get('outputTokens', 'N/A')}")

    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("Note: Each API call includes the FULL conversation history.")

if __name__ == "__main__":
    multi_turn_conversation()
