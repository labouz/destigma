# the script for labeling a post as stigma or not, as drug related or noti 
import pandas as pd
import pickle
import os
import time
import google.generativeai as genai

def get_stigma_drug(post, retries = 2, api_key = None):
# prompt
    prompt = f"""
    **Instructions:**\
    You are an expert sociologist on stigma among people who use drugs. You have been asked to label posts from various Reddit communities on the disclosure of a stigma utterance.\
    1. Provide one label for each post: S-D, S-ND, NS-D, NS-ND. If you are unsure, provide your best guess. \
    2. If the post is labeled S-D or S-ND, extract the exact utterance of stigma and provide a brief description of the stigma.\

    **Definitions:**\
    The inclusion of stigma can be explicit or implicit and can be directed at an individual or a group using stigmatizing language or stereotypes.\
    Stigma usually includes four intrinsic features: marks, labels, etiology, and peril. Marks are nonverbal cues that identify members of a stigmatized group. Labels arouse and reflect social cognitions, such as considering stigmatized people to be a distinct group, highlighting intergroup differences, encouraging categorization, and promoting stereotypes. Etiology content describes stigmatized people’s voluntary decisions to violate their social responsibilities and engage in taboo activities. Peril content links stigmatized people to physical, neurological, and social threats, such as pain, death, aggression, and taboo behavior\
    The post author can describe an instance where they felt stigmatized by another person or a system or describe feelings of shame or guilt because of their drug use.\
    If the post author describes a situation where they feel unsure or worried because of how they will be perceived by others, anticipating being discrimminated against because of their membership in the group, this can also be considered a post about stigma.\
    You can label a post in one of four ways: \
    1. S-D: The post contains an utterance of stigma. Either language that is stigmatizing or a description of an experience related to stigma *AND* the stigma is related to drug use.\
    2. S-ND: The post contains an utterance of stigma. Either language that is stigmatizing or a description of an experience related to stigma *AND* the stigma is *NOT* related to drug use.\
    3. NS-D: The post does *NOT* contain any utterance of stigma *AND* the post *IS* related to drug use.\
    4. NS-ND: The post does *NOT* contain any utterance of stigma *AND* the post is *NOT* related to drug use.\

    **Considerations:**\
    1. Sometimes people use coded language to refer to their drug use (e.g. 'stoner' to refer to being high on cannabis, 'ents' for a person who enjoys cannabis, or '[5]' to refer to the degree of the user's high out of 10) \
    2. Focus on Intent: Stigmatizing language aims to degrade, shame, or perpetuate negative stereotypes about people who use drugs. \
    3. Context Matters: Consider the overall tone and purpose of the post. Is it intended to inform, share an experience, or genuinely promote negativity towards people who use drugs?\

    **Examples:**\
    """

    # few-shot examples
    example1 = "You did it. Take responsibility for it, US Government. We didn't want to break the law. We only wanted to do something harmless and fun. But we got used to the idea of breaking the law on a regular basis. We learned to be afraid of the police.You taught us that the justice system is not our friend.And so we got very comfortable living outside the law."
    answer1 = "S-D, 'Take responsibility for it, US Government. We didn't want to break the law. We only wanted to do something harmless and fun. But we got used to the idea of breaking the law on a regular basis. We learned to be afraid of the police.You taught us that the justice system is not our friend.And so we got very comfortable living outside the law.', The author is describing a situation where they feel there are systemic barriers that prevent them from living a normal life and are blaming the US government for their situation. The author is also describing a situation where they feel they are being discriminated against because of their drug use."
    example2 = "I hope my junkie sister OD's or disappears out of our lives My sister is an alcoholic junkie who has 2 DUIs under her belt as well as loves taking Xanax and alcohol together and wreaking havoc for our family and even strangers."
    answer2 = "S-D, 'junkie', The author is using the term junkie to describe their sister and is perpetuating the stereotype that people who use drugs are dangerous and harmful."
    example3 = "Moral of the story is never trust a low life no matter how much you feed them and house them and be friendly with them, at the end of the day they don't have shit and if you have shit they're gonna fucking take it."
    answer3 = "S-ND, 'never trust a low life', The author is labeling a person who is unhoused as a 'low-life' and is perpetuating the stereotype that they are thieves and including a message of peril that your belongings will get stolen if you choose to support them."
    example4 = "My mom is going to kick me out. She graciously gave me the choice of getting dropped off in a shelter in either San Diego or the desert area (Palm Springs and surrounding areas). I would choose the desert because that is one of my old stomping grounds. The dope is phenomenal and cheap (3g's for $100) and the homeless population is a majority young people. I can also hustle up $350 and rent a room at a buddy's place. I have a few options that I can look at but I have to figure it out soon."
    answer4 = "NS-D"
    example5 = "So, I got in touch with a friend who grows. I smoked some weed, and I never felt better. Not the high, but after. After the high I was finally happy. Finally. My anxiety was gone and I never felt better. So I decide to smoke for two months then tell my mom and get my green card. She said I seem happier and stuff, then I told her why and she flipped."
    answer5 = "NS-D"
    example6 = "I think I genuinely may have developed some sort of PTSD due to the incident. I have had an anxiety disorder of some kind for my entire life but over the past few years it has been getting steadily worse and I wonder if it is due to that incident."
    answer6 = "NS-ND"
    example7 = "Death with dignity people are wrong. I know the argument that people will be in lots of pain near the end of their life so they should kill themselves early on when they get cancer or whatever. I don't think it's very dignified to kill yourself to avoid temporal pain. It also begs the question of what is considered enough pain or chance of not getting better is enough be think killing yourself is a good idea. I've been in extreme pain before, sometimes for extended periods of time, but I didn't think killing myself was the best solution to my problems."
    answer7 = "S-ND, 'I don't think it's very dignified to kill yourself to avoid temporal pain', The author is insenuating that people with a terminal disease have no dignity when they choose a compassionate death and is assigning blame to that person."
    example8 = "When I'm dating a solid 10/10 girl there is close to nothing she can do wrong during step 2 that won't still merit her the joys of step 3. I, on the other hand, can kill the whole thing deader than Hitler by making one tiny mistake. Perhaps I'll text her back too quickly giving her the impression I'm desperate. Well princesses can't we all be equal and stop this stupid fucking game? No you say? Well why not? Men everywhere have to suffer your shit like slaves to pussy! But it's obvious why it must be this way; because 10/10s don't date guys they want to fuck! Why would they? Cock runs from the taps for these princesses! Wait, that's it... Don't date 10/10s."
    answer8 = "S-ND, 'I'm dating a solid 10/10 girl; princesses; because 10/10s don't date guys they want to fuck', The author is labeling attractive women as princesses and perpetuating a stereotype. They are also issuing a warning about dating attractive women."

    combined_prompt = prompt + "\n\n" + "\n\n".join([
        example1 + "\nAnswer: " + answer1,
        example2 + "\nAnswer: " + answer2,
        example3 + "\nAnswer: " + answer3,
        example4 + "\nAnswer: " + answer4,
        example5 + "\nAnswer: " + answer5,
        example6 + "\nAnswer: " + answer6,
        example7 + "\nAnswer: " + answer7,
        example8 + "\nAnswer: " + answer8,
        post
    ])
    genai.configure(api_key= api_key)
    model = genai.GenerativeModel('gemini-pro')

    while retries > 0:
        try:
            response = model.generate_content(
                combined_prompt
            )
            label = response.text.lower().strip()  # Extract label from response
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

