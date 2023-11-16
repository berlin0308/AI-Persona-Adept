# AI Persona Adept: Your Personalized Consultant

## Introduction
This project is a Personality Assessment Application developed using **OpenAI API**, designed to provide a more personalized assessment of the user's personality, offering an engaging and valuable personality evaluation.

## Repository Structure

+ **'pa_output/'** : output files that are generated by the project. 
  - **state/** : save files of the runs
  - **summary/** : summary of the runs
+ **'static/'** : static files that are used by the project.
  - **question_list/** : questions and scoring critiria
  - **user_prompt/** : prompt for users
+ **'pa_config.json'** : configuration file that includes all the necessary settings and parameters required.
+ **'persona_adept.py'** : main program of the project.

## Description

The Assessment consists of four stages:
+ **Multiple Choice:** Users provide their preferences by submitting scores from 1 to 5 based on the question descriptions.
+ **Open-ended Questions:** Users provide textual responses, and the system may follow up for clarification if necessary.
+ **Adaptive Questions:** The system adapts its queries based on previous responses, focusing on areas that are less clear.
+ **End and Summarize:** The system considers all previous questions and responses along with an internal scoring system, and identifies the user's MBTI personality type and offers a comprehensive analysis.

## LLM Task Details
+ **Natural Language Understanding and Scoring**
  - Utilizing GPT-3.5 to comprehend the semantic meaning of user responses
  - Able to estimate scores based on the predefined [scoring criteria](https://github.com/berlin0308/AI-Persona-Adept/blob/main/static/question_list/scoring_prompt_first.txt).
  - For responses that are unclear, ask followup question and score based on the [followup scoring criteria](https://github.com/berlin0308/AI-Persona-Adept/blob/main/static/question_list/scoring_prompt_followup.txt)
 
+ **Response Format Assertion**
  - Employ *assert_format()* function to validate the correctness of the format.
    ``` python
    def assert_format(self, text, format, N=False):
        if format == 'score_-5_to_5':
            try:
                if int(text) >= -5 and int(text) <= 5:
                    return True
                else:
                    return False 
            except:
                if N and str(text) == 'N':
                    return True
                return False
        elif format == 'MBTI':
            if len(format) == 4:
                return True
            return False
    ```
  - Rerun the process in cases where the response generated by GPT does not conform to the prescribed format.
    ``` python
    score = self.gpt.run_gpt(gpt_in, 200)
    while not self.assert_format(score, 'score_-5_to_5', N=True):
      time.sleep(2)
      score = self.gpt.run_gpt(gpt_in, 200)
    ```

+ **Summarization**
  - During the assessment, all questions and responses are recorded within the 'State' class.
  - A personalized personality assessment report is generated with the [specified article structure](https://github.com/berlin0308/AI-Persona-Adept/blob/main/static/question_list/summary_instruction.txt).
  - Example results of summarization are [Here](https://github.com/berlin0308/AI-Persona-Adept/tree/main/pa_output/summary)

## **Example Runs**
  - Run-1: An ISTP person with [the process](https://github.com/berlin0308/AI-Persona-Adept/blob/main/pa_output/state/save_1.json) and [the result](https://github.com/berlin0308/AI-Persona-Adept/blob/main/pa_output/summary/summary_1.txt)
  - Run-2: An ESTJ person with [the process](https://github.com/berlin0308/AI-Persona-Adept/blob/main/pa_output/state/save_2.json) and [the result](https://github.com/berlin0308/AI-Persona-Adept/blob/main/pa_output/summary/summary_2.txt)
  - Run-3: An INTP person with [the process](https://github.com/berlin0308/AI-Persona-Adept/blob/main/pa_output/state/save_3.json) and [the result](https://github.com/berlin0308/AI-Persona-Adept/blob/main/pa_output/summary/summary_3.txt)

## **My discovery**


vd
