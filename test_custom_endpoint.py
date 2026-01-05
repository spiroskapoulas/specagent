#!/usr/bin/env python3
"""
Test script for custom inference endpoint.

Usage:
    python test_custom_endpoint.py
"""

from specagent.llm.custom_endpoint import create_custom_llm
from specagent.nodes.router import ROUTER_PROMPT, RouteDecision
import json
import re

def test_router():
    """Test router node with custom endpoint."""
    print("Testing Router with Custom Endpoint")
    print("=" * 70)

    # Create LLM client
    endpoint_url = "http://qwen3-4b-predictor.ml-serving.10.0.1.2.sslip.io:30750/v1/chat/completions"
    llm = create_custom_llm(endpoint_url=endpoint_url)

    # Test question
    question = "What is the maximum number of HARQ processes in 5G NR?"
    prompt = ROUTER_PROMPT.format(question=question)

    print(f"Question: {question}\n")
    print("Calling endpoint...")

    try:
        # Call LLM
        response = llm.invoke(prompt)
        print(f"\n✓ Response received:")
        print(response)

        # Parse JSON
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            decision = RouteDecision(**parsed)

            print(f"\n✓ Parsed decision:")
            print(f"  Route: {decision.route}")
            print(f"  Reasoning: {decision.reasoning}")

            if decision.route == "retrieve":
                print(f"\n✅ SUCCESS! Router correctly identified 3GPP question")
            else:
                print(f"\n⚠ Router rejected the question (unexpected)")
        else:
            print(f"\n⚠ Response doesn't contain JSON")
            print("   You may need to adjust the prompt")

    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check endpoint is accessible: curl " + endpoint_url)
        print("  2. Verify firewall/network allows connections")
        print("  3. Check service logs for errors")

    print("=" * 70)


def test_simple():
    """Simple test to verify endpoint connectivity."""
    print("\nSimple Connectivity Test")
    print("=" * 70)

    endpoint_url = "http://qwen3-4b-predictor.ml-serving.10.0.1.2.sslip.io:30750/v1/chat/completions"
    llm = create_custom_llm(endpoint_url=endpoint_url)

    try:
        response = llm.invoke("Say hello in one word")
        print(f"✓ Response: {response}")
        print("\n✅ Endpoint is working!")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("=" * 70)


if __name__ == "__main__":
    # Run simple test first
    test_simple()

    # Then test router
    test_router()
