import openai
import json
import re

openai.api_key = "<YOUR_API_KEY>"

def generate_question(prompt):
    print("Generating question...")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=30,
        # n=1,
        # stop=None,
        temperature=0.5,
    )
    print("Question generated!")
    question = response.choices[0].text.strip()
    return question

def evaluate_answer(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=10,
        # n=1,
        # stop=None,
        temperature=0.5,
    )
    text = response.choices[0].text.strip()
    score = int(re.findall(r'\d+', text)[0])
    return score

def analyze_strengths_weaknesses(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5,
    )
    analysis = response.choices[0].text.strip()
    return analysis

software_engineering_topics = [
    # "data structures",
    # "algorithms",
    # "object-oriented programming",
    # "operating systems",
    # "databases",
    #"Data Engineering",
    "Spark",
    "Spark Job optmization"
    #"AWS"
]

interview_score = 0
total_questions = 1
strengths_weaknesses = []
difficulty = "medium"
interview_context = ""

print("AI Interviewer: Software Engineering Questions")
print("----------------------------------------------")

for i, topic in enumerate(software_engineering_topics[:total_questions]):
    question_prompt = f"Generate a commonly asked interview question about {topic} that you haven't asked me before & is of {difficulty} difficulty."
    question = generate_question(question_prompt)
    print(f"Question {i + 1}: {question}")

    candidate_answer = input("Your answer: ")
    interview_context += f"Question {i + 1}: {question}\nAnswer {i + 1}: {candidate_answer}\n"

    evaluation_prompt = f"Evaluate the quality of the following answer to the question '{question}': {candidate_answer}. Give a score between 0 (worst) and 10 (best)."
    score = int(evaluate_answer(evaluation_prompt))
    interview_score += score
    print(f"Score: {score}\n")

    analysis_prompt = f"Analyze the strengths and weaknesses in the following answer to the question '{question}': {candidate_answer}."
    analysis = analyze_strengths_weaknesses(analysis_prompt)
    strengths_weaknesses.append(analysis)
    print(f"Analysis:\n {analysis}\n")

# Generate cumulative strength and weakness analysis
cumulative_analysis_prompt = f"Based on the answers provided by the candidate during the interview, provide a cumulative analysis of their overall strengths and weaknesses. The interview context is as follows:\n{interview_context}"
cumulative_analysis = analyze_strengths_weaknesses(cumulative_analysis_prompt)

print("Interview Summary")
print("-----------------")
print(f"Total Score: {interview_score} / {total_questions * 10} \n")

# for i, sw in enumerate(strengths_weaknesses):
#     print(f"Question {i + 1}: {software_engineering_topics[i]}")
#     print(f"Strengths & Weaknesses:\n {sw}\n")

print("Cumulative Strengths & Weaknesses")
print("---------------------------------")
print(cumulative_analysis)