# the script for labeling a post as durg-related or not
import pandas as pd
import openai
import time
from threading import Lock

rate_limit = 10000
tpm_limit = 1000000  # Tokens per minute
rate_limit_period = 60  # seconds
retry_wait_time = 5  # seconds between retries

# Global variables for tracking rate limit
request_count = 0
token_count = 0
request_lock = Lock()
token_lock = Lock()

def get_drug_post(post, retries = 2, model = None, openai_client=None):
# prompt
    prompt = f"""
    *Instructions for Labeling Drug References in Social Media Posts*

    1. **Objective**: Identify references to drugs or people who use drugs in each post.

    2. **Include**:
    - Illicit Drugs: All controlled substances with no legal usage (e.g. cannabis, heroin, cocaine, methamphetamine)
    - Prescription Drugs: drugs that are often abused even if they have legitimate medical uses (e.g., opioids, benzodiazepines).
    - Other Drugs: drugs that are non-prescription and known to be abused (e.g., inhalants, k2, bath salts).
    - Explicit mentions of drug use, abuse, or addiction related terms (e.g., "getting high", "stoned").

    3. **Exclude**:
    - Tobacco, nicotine, or alcohol unless explicitly linked to drug use.
    - Do not include medical or psychological discussions unless there is a direct and clear mention of drug use or abuse.

    4. **Clarifications**:
    - Mental health discussions should not be labeled as 'D' unless there is an explicit mention of drugs as defined above.
    - Use 'ND' for posts that discuss health or psychological issues without specific drug references.

    5. **Language Cues**:
    - Focus on clear drug-related terminology (e.g., "junkie", "addict") and slang.
    - If a post is ambiguous and does not clearly fit the drug reference criteria, label as 'ND'.

    6. **Response Requirement**:
    - Respond with either 'D' (Drug) or 'ND' (Non-Drug) based on these guidelines. No additional commentary is needed.
    """
#     prompt = f"""
#     *Instructions for Labeling Drug References in Social Media Posts*

#     1. **Task Overview**: You are tasked with identifying any references to drugs or people who use drugs within each post. 

#     2. **Substances Included**:
#     - **Illicit Drugs**: Consider all controlled substances with no legal usage.
#     - **Prescription Drugs**: Include drugs that are often abused even if they have legitimate medical uses.
#     - **Other Substances**: Include any non-prescription substances known to be abused (e.g., inhalants, synthetic compounds).
#     - **Exclude**: Do not consider tobacco, nicotine, or alcohol unless explicitly linked to drug use or addiction.

#     3. **Drug Classes**:
#     - **Narcotics**, **Stimulants**, **Depressants**, **Hallucinogens**
#     - **Cannabis**, **Drugs of Concern** (e.g., DXM, fake pills, kratom)
#     - **Designer Drugs** (e.g., bath salts, spice, K2)
#     - **Treatment Substances**: Include methadone and other synthetic opiates used in treatment contexts.

#     4. **Language Cues**:
#     - Pay attention to slang and euphemisms for drugs.
#     - Phrases indicating drug use effects, such as "stoned" or "high", should lead to a 'D' label.
#     - The use of names referring to people who use drugs (e.g., junkie, addict) should also be considered as 'D'.
#     - Be cautious with ambiguous references; if in doubt, label as 'ND'.

#     5. **Classification**:
#     - **D**: Label the post as 'D' if it discusses drugs, drug use, or people who use drugs, including any of the aforementioned substances and contexts.
#     - **ND**: Label the post as 'ND' if it does not discuss drugs or drug use, and contains no references to the substances outlined above.

#     6. *Clarification on Psychological References*:
#     - Do not automatically classify posts as 'D' when they mention medical or psychological assessments, doctors, or treatments, unless there is a direct reference to drug use or medications known to be abused.
#     - Be cautious with posts discussing mental health issues; label as 'D' only if there are explicit mentions of substances covered under the drug categories specified in the instructions.
#     - Do not assume that mental health discussions imply drug use.
    
#    7.  *Response Requirement*:
#     - Respond with a single label per post: either 'D' or 'ND'. Nothing more.

#     """
    example1 = "I'm so high right now, I can't even feel my face. This is the best weed I've ever smoked."
    answer1 = "D"
    example2 = "I hope my junkie sister OD's or disappears out of our lives My sister is an alcoholic junkie who has 2 DUIs under her belt as well as loves taking Xanax and alcohol together and wreaking havoc for our family and even strangers."
    answer2 = "D"
    example3 = "My mom is going to kick me out. She graciously gave me the choice of getting dropped off in a shelter in either San Diego or the desert area (Palm Springs and surrounding areas). I would choose the desert because that is one of my old stomping grounds. The dope is phenomenal and cheap (3g's for $100) and the homeless population is a majority young people. I can also hustle up $350 and rent a room at a buddy's place. I have a few options that I can look at but I have to figure it out soon."
    answer3 = "D"
    example4 = "I cut tonight I cut myself tonight after I've been clean for a few months now and I'm honestly just broken inside. I had a huge fight with my girlfriend, I'm worried we might break up. I just kinda lost control, grabbed my razorblade and cut. I feel like crap and am honestly just lost right now."
    answer4 = "ND"
    example5 = "My drinking problem is way worse than anyone knows. I'm an alcoholic and the people around me know that, because I told them. But if I hadn't, they wouldn't have guessed, judging by their reactions of shock and disbelief and the actual ‚are you sure you're not exaggerating, that I got when I came clean a couple years ago."
    answer5 = "ND"
    example6 = "My mother died two years ago from cancer of basically all of her internal organs starting in the colon and then liver and spleen and just everywhere. I hadn't seen her in years because she struggled with drug addiction since I was about four or five. We had years together but not enough. I was there the day she died and I wouldn't wish anyone to go through what she did or what our family did when we saw her in that state. My father also had an addiction to pain pills and alcohol. My last few years of high school were ruined because of that. I watched him overdoes on thanksgiving. I watched him try to kill himself with his own hand gun more than once. I watched him aim it at my stepmom when she tried to take it from him. I heard him yell at me while he kicked me and told me that he didn't want me anymore and tried to drag me up the stairs. Maybe these are some of the reasons I drink. Maybe because I barely have a relationship with my dad now and he still hasn't said anything to me today."
    answer6 = "D"
    example7 = "Hate my new job I just got hired at Taco Bell a few weeks ago and I really hate it. The only thing that's gets me through it is smoking weed. I wonder if i should just quit or no call no show?"
    answer7 = "D"
    example8 = "I'm trying not to get emotionally invested again in you because I know all to well how you are. You're an alcoholic and I know there is nothing I can do to save you. It kills me to see you drink like you do but I know how it is, you have to want to stop. As much as I love fucking you and being around you, It's not a good idea to continue being around you. "
    answer8 = "ND"
    example9 = "Recently I took a psychological exam for work. To know if I'm fit to work.The doctor looked at my paper and asked 'Do your have urges to hurt/killpeople?' I laughed in my head and lied.."
    answer9 = "ND"

    global request_count, token_count

    while retries > 0:
        try:
            with request_lock:
                if request_count >= rate_limit:
                    print("Rate limit reached. Pausing for a minute...")
                    time.sleep(rate_limit_period)
                    request_count = 0

            with token_lock:
                if token_count >= tpm_limit:
                    print("TPM limit reached. Pausing for a minute...")
                    time.sleep(rate_limit_period)
                    token_count = 0


            response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": example1

                },
                {
                    "role": "system",
                    "content": answer1
                },
                {
                    "role": "user",
                    "content": example2
                },
                {
                    "role": "system",
                    "content": answer2
                },
                {
                    "role": "user",
                    "content": example3
                },
                {
                    "role": "system",
                    "content": answer3
                },
                {
                    "role": "user",
                    "content": example4
                },
                {
                    "role": "system",
                    "content": answer4
                },
                {
                    "role": "user",
                    "content": example5
                },
                {
                    "role": "system",
                    "content": answer5
                },
                {
                    "role": "user",
                    "content": example6
                },
                {
                    "role": "system",
                    "content": answer6
                },
                {
                    "role": "user",
                    "content": example7
                },
                {
                    "role": "system",
                    "content": answer7
                },
                {
                    "role": "user",
                    "content": example8
                },
                {
                    "role": "system",
                    "content": answer8
                },
                {
                    "role": "user",
                    "content": example9
                },
                {
                    "role": "system",
                    "content": answer9
                },
                {
                    "role": "user",
                    "content": post
                }
            ],
            model=model,
            temperature=0
        )
            with request_lock:
                request_count += 1

            with token_lock:
                token_count += sum([len(prompt), len(response.choices[0].message.content)])

            label = response.choices[0].message.content.lower().strip()
            return label
        except openai.RateLimitError as e:
            print(f"Rate limit error: {e}. Retrying in {retry_wait_time} seconds...")
            time.sleep(retry_wait_time)
            retries -= 1
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            retries -= 1
            time.sleep(retry_wait_time)

    print("Retrying...")
    return "skipped"

