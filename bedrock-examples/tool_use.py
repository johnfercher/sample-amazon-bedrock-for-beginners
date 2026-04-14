import json
import boto3

# ---------------------------------------------------------------------------
# Step 1: Define your local Python functions
# ---------------------------------------------------------------------------
# These are regular Python functions. The model will never call them directly.
# Instead, the model will ASK us to call them by returning a tool_use block.

def get_weather(location, unit="fahrenheit"):
    """
    Simulate fetching weather data for a location.
    In a real app, this would call a weather API like OpenWeatherMap.
    """
    # Fake weather data for the demo
    weather_data = {
        "location": location,
        "temperature": 58 if unit == "fahrenheit" else 14,
        "unit": unit,
        "condition": "Partly cloudy",
        "humidity": "72%",
        "wind": "8 mph NW",
    }
    return weather_data


# ---------------------------------------------------------------------------
# Step 2: Describe your functions as "tools" for the model
# ---------------------------------------------------------------------------
# The model needs a description of each tool so it knows:
#   - What the tool does (description)
#   - What inputs it expects (inputSchema)
#
# This is like writing documentation so someone else can use your function.

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get the current weather for a given location.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. 'San Francisco, CA'",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["fahrenheit", "celsius"],
                                "description": "Temperature unit (default: fahrenheit)",
                            },
                        },
                        "required": ["location"],
                    }
                },
            }
        }
    ]
}


# ---------------------------------------------------------------------------
# Step 3: Map tool names to actual Python functions
# ---------------------------------------------------------------------------
# When the model asks to use a tool, we look up the function by name here.

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
}


def run_tool(tool_name, tool_input):
    """
    Look up a tool by name and call it with the provided input.
    Returns the result as a dictionary.
    """
    func = TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        return {"error": f"Unknown tool: {tool_name}"}

    # ** unpacks the dict into keyword arguments:
    #   get_weather(**{"location": "Seattle"})  →  get_weather(location="Seattle")
    return func(**tool_input)


# ---------------------------------------------------------------------------
# Step 4: The main tool use loop
# ---------------------------------------------------------------------------

def tool_use_demo():
    # Create the Bedrock client
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "us.amazon.nova-lite-v1:0"

    user_message = "What's the weather like in Seattle right now?"

    print("Bedrock Tool Use Demo")
    print("=" * 60)
    print(f"User: {user_message}\n")

    # Start the conversation with the user's message
    messages = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    # --- First API call ---
    # Send the message AND the tool definitions to the model.
    # The model will look at the question, look at the available tools,
    # and decide if it needs to call one.
    print("[Step 1] Sending message to model with tool definitions...")

    response = bedrock.converse(
        modelId=model_id,
        messages=messages,
        toolConfig=TOOL_CONFIG,
        inferenceConfig={"temperature": 0.0, "maxTokens": 300},
    )

    stop_reason = response["stopReason"]
    assistant_message = response["output"]["message"]

    print(f"  Model responded with stop reason: {stop_reason}")

    # --- Check: did the model ask to use a tool? ---
    if stop_reason == "tool_use":
        # The model wants to call a tool. Let's find the toolUse block.
        tool_use_block = None
        for block in assistant_message["content"]:
            if "toolUse" in block:
                tool_use_block = block["toolUse"]
                break

        tool_name = tool_use_block["name"]
        tool_input = tool_use_block["input"]
        tool_use_id = tool_use_block["toolUseId"]

        print(f"\n[Step 2] Model wants to call: {tool_name}")
        print(f"  With arguments: {json.dumps(tool_input, indent=2)}")

        # --- Run the actual function ---
        result = run_tool(tool_name, tool_input)
        print(f"\n[Step 3] Function returned: {json.dumps(result, indent=2)}")

        # --- Send the result back to the model ---
        # We add the assistant's message (with the tool request) to the history,
        # then add a user message containing the tool result.
        messages.append(assistant_message)
        messages.append({
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"json": result}],
                    }
                }
            ],
        })

        print("\n[Step 4] Sending tool result back to model...")

        final_response = bedrock.converse(
            modelId=model_id,
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"temperature": 0.0, "maxTokens": 300},
        )

        final_text = final_response["output"]["message"]["content"][0]["text"]
        print(f"\nAssistant: {final_text}")

    elif stop_reason == "end_turn":
        # The model answered directly without needing a tool
        print(f"\nAssistant: {assistant_message['content'][0]['text']}")


if __name__ == "__main__":
    tool_use_demo()
