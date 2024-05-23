# the script for labeling a post as experience, othering, or self-stigma 
import pandas as pd
import pickle
import os
import time
import openai

def get_utterance(post, retries = 2, model = "gpt-4-0125-preview", openai_client=None):
    prompt = f"""
    **Instructions:**\
    You are an expert sociologist on the stigma faced by people who use drugs. You have been given a set of posts from various Reddit communities that have been labeled as containing an utterance of stigma related to drug use.\
    The utterance can be in the form of direct stigmatizing language towards drug use and people who use drugs, a person who uses drugs disclosing an experience where they felt stigmatized against, or language from the person who uses drugs that endorses negative stereotypes towards themselves and is internalized. \
    Your task is to determine if the post contains stigmatizing language, description of a stigmatizing experience related to drug use, or self-stigma.\
    Sometimes a post may contain two or more types of stigma utterance, in that case, carefully choose the type that is directly related to their drug use.\
    If the post does not contain any stigma directly related to drug use, mark it as non-related.\
    **Definitions:**\
    * Othering: Words or phrases that are intended to degrade, shame, or perpetuate negative stereotypes about people who use drugs. Usually the subject of the phrase is the stigmatized person(s) and image or descriptor as predicate nominative. Stigmatizing language often includes a mark (cues that classify a person as part of a stigmatized group), label (reinforcement of 'us' versus 'them' mentality), and/or etiology/responsibility. \
    * Experience: A description of an instance where the post author felt stigmatized by another person or a system. Often they themselves are the subject of the sentence when speaking, when recounting an experience that has happened in the past.\
    * Self-stigma: Language that endorses negative stereotypes towards themselves and is internalized. Often they themselves are the subject of the sentence when speaking, often directly stating something they believe about themselves. Self-stigma could also include the anticipation that someone who does drugs will be judged or be falsly accused of something. \
    * Non-related: the stigma is not related to drug use. \
    **Considerations:**\
    1. Context: Consider the overall tone and purpose of the post. Is it intended to inform, share an experience, or genuinely promote negativity towards people who use drugs?\
    2. Focus on Intent: Stigmatizing language aims to degrade, shame, or perpetuate negative stereotypes about people who use drugs. \
    3. Please consider if the author seems to be speaking from personal experience or if they are speaking about others. Meaning, are they in the community they are speaking about (in-group) or are they speaking about a community they are not a part of (out-group)?\
    Mark the post as either: Othering, Experience, or Self. If you are unsure, provide your best guess.
    """
    # examples
    example1 = "People who use drugs are a burden on society. They should be locked up and never let out."
    answer1 = "Othering"
    example2 = "I was walking down the street and someone called me a junkie. I felt so ashamed."
    answer2 = "Self"
    example3 = "Hard truth to swallow: If a drug ruined your life, then you just sucked at doing it Plenty of people have used even the hardest drugs and never ruined their life. The addiction rate of drugs is far lower than most realize. (example, 25p of people who try heroin become addicts) There are proper ways to use drugs. Moderation and self control is everything and its not the drug that ruins people's lives, it's the poor lifestyle choices they start making."
    answer3 = "Othering"
    example4 = "So basically a friend of mine who we think is addicted to coke came over and did some. Another friend decided to take the remainder away before she could do more to help her.He is withholding it from her, so now she's going nuts because of withdrawal, and is doing anything to get it back (threatening to suicide, etc). My question is, how long will her craziness last? "
    answer4 = "Othering"
    example5 = "AMC fires cancer patient for testing positive for marijuana. So, my fiancee smokes marijuana for medical purposes... he has  leiomyosarcoma (albeit illegally because Ohio and Kentucky suck when it comes to medical marijuana) and got fired for testing positive.  You're probably thinking that he should have known the risks, blah blah blah... but it wasn't a random drug test."
    answer5 = "Experience"
    example6 = "I hate the misconception most people have about being high on Marihuana.When people (non-drug users) imagine someone being high on weed they think of it as a person who is really fucked up and can't function or do anything, something most people believe it's real when it comes substances referred to as drugs (ironic because alcohol is also a drug, anyways...) and it pisses me off because it makes them seem ignorant."
    answer6 = "Experience"
    example7 = "I've been on Xanax for a year or so and I'd be lying if I said I didn't abuse it.I recently met a girl who told me once that she can't really give a fuck about mental health needs. I'm just very afraid that she won't accept this part of me when the time comes and I'm having an attack."
    answer7 = "Self"
    while retries > 0:
        try:
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
                    "content": post
                }

            ],
            model=model,
            temperature=0
        )
            label = response.choices[0].message.content.lower().strip()
            return label
        except Exception as e:
            if e:
                print(e)
                retries -= 1
                time.sleep(5)
            else:
                raise e
    print("Retrying...")
    return "skipped"