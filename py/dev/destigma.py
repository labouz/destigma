import pandas as pd
import pickle
import time
from openai import OpenAI
import asyncio
from collections import deque
from task1_drug import get_drug_post_async

# 1,5 million
non_drug_sub = pickle.load(open('../data/non_drug_data_filtered_delauthor.pkl', 'rb'))
seed = 5
dat = non_drug_sub.sample(50, random_state = seed)
del non_drug_sub
api_key = "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"
client = OpenAI(api_key = api_key)
############################################################################################################
# TASK1 - is the post about drugs or not
async def process_posts(posts, batch_size=10, concurrency=5):
    openai_client = client  # Initialize client once
    results = []
    batch_queue = deque(posts)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_batch():
        nonlocal batch_queue, results
        async with semaphore:
            while batch_queue:
                batch = [batch_queue.popleft() for _ in range(batch_size) if batch_queue]
                batch_tasks = [get_drug_post_async(post, openai_client=openai_client) for post in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)

    await asyncio.gather(*[process_batch() for _ in range(concurrency)])  # Start multiple batch processors
    return results

# start = time.time()
results = asyncio.run(process_posts(dat['text'].tolist()))
# end = time.time()

# print(f"Time taken: {end - start} seconds")

# Save results
results_df = pd.DataFrame({'text': dat['text'], 'label': results})
results_df.to_csv('../data/task1_results.csv', index=False)
print(results_df.head())