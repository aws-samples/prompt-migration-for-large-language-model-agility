class FaithfulnessTemplate:
    @staticmethod
    def generate_claims(text):
        return f"""
Human: 

<instruction>
Based on the given text, please generate a comphrensive list of FACTUAL claims that can inferred from the provided text.
</instruction>

<example>
<text>
"Einstein won the noble prize in 1968 for his discovery of the photoelectric effect."
</text>

<generated_json> 
{{
    "claims": [
        "Einstein won the noble prize for his discovery of the photoelectric effect.",
        "Einstein won the noble prize in 1968."
    ]  
}}
</generated_json>
</example>

**
IMPORTANT: Please make sure to only return in JSON format, with the "claims" key as a list of strings. No words or explaination is needed.
Only include claims that are factual, and the claims you extract should include the full context it was presented in, NOT cherry picked facts.
You should NOT include any prior knowledge, and take the text at face value when extracting claims.
**

Now here is the actual text:
<text>
{text}
</text>

Please only output the generated json and nothing else. 

Assistant: 
Here is the generated json: 
"""

    @staticmethod
    def generate_truths(text):
        return f"""
Human: 
<instruction>
Based on the given text, please generate a comphrensive list of FACTUAL, undisputed truths that can inferred from the provided text.
</instruction>

<example>
<text>
"Einstein won the noble prize in 1968 for his discovery of the photoelectric effect."
</text>

<generated_json>
{{
    "truths": [
        "Einstein won the noble prize for his discovery of the photoelectric effect.",
        "Einstein won the noble prize in 1968."
    ]  
}}
</generated_json>
</example>

**
IMPORTANT: Please make sure to only return in JSON format, with the "truths" key as a list of strings. No words or explaination is needed.
Only include truths that are factual.
**

Now here is the actual text:

<text>
{text}
</text>

Please only output the generated json and nothing else. 

Assistant:
Here is the generated json:
"""

    @staticmethod
    def generate_verdicts(claims, retrieval_context):
        return f"""
Human: 
<instruction>
Based on the given claims, which is a list of strings, generate a list of JSON objects to indicate whether EACH claim contradicts any facts in the context. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes', 'no', or 'idk', which states whether the given claim agrees with the context. 
Provide a 'reason' ONLY if the answer is 'no'. 
The provided claim is drawn from the actual output. Try to provide a correction in the reason using the facts in the context.

IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
</instruction>

<example>
<contexts> 
"Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
</contexts>

<claims>
["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a Germen chef."]
</claims>

<generated_json>
{{
    "verdicts": [
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein won the Nobel Prize in 1969, which is untrue as the context states it is 1968 instead."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein is a Germen chef, which is not correct as the context states he was a German scientist instead."
        }},
    ]  
}}
</generated_json>
</example>

The length of 'verdicts' SHOULD BE STRICTLY EQUAL to that of claims.
You DON'T have to provide a reason if the answer is 'yes' or 'idk'.
ONLY provide a 'no' answer if the context DIRECTLY CONTRADICTS the claims. YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
Claims made using vague, suggestive, speculative language such as 'may have', 'possibility due to', does NOT count as a contradiction.
Claims that is not backed up due to a lack of information/is not mentioned in the contexts MUST be answered 'idk', otherwise I WILL DIE.

Now here are the actual contexts and claims:

<contexts>
{retrieval_context}
</contexts>

<claims>
{claims}
</claims>

Please only output the generated json and nothing else. 

Assistant: 
Here is the generated json:
"""

    @staticmethod
    def generate_reason(score, contradictions):
        return f"""
Human: 
<instruction>
Below is a list of Contradictions. It is a list of strings explaining why the 'actual output' does not align with the information presented in the 'context'. Contradictions happen in the 'actual output', NOT the 'context'.
Given the faithfulness score, which is a 0-1 score indicating how faithful the `actual output` is to the context (higher the better), CONCISELY summarize the contradictions to justify the score. 

<faithfulness_score>
{score}
</faithfulness_score>

<contradictions>
{contradictions}
</contradictions>

<output_format>
The score is <faithfulness_score> because <your_reason>.

**
IMPORTANT: 
If there are no contradictions, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
Your reason MUST use information in `contradiction` in your reason.
Be sure in your reason, as if you know what the actual output is from the contradictions.
**

Assistant: 
Here is your response based on the output format: 
"""