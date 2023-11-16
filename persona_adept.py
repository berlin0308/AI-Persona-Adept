import os
import re
import sys
import json
import random
import logging
import argparse
import time

import openai
import tiktoken
from requests.exceptions import Timeout


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(funcName)s() - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


def read_txt(file, write_log=False):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        text = f.read()

    if write_log:
        characters = len(text)
        logger.info(f"Read {characters:,} characters")
    return text


def write_txt(file, text, write_log=False):
    if write_log:
        characters = len(text)
        logger.info(f"Writing {characters:,} characters to {file}")

    with open(file, "w", encoding="utf8") as f:
        f.write(text)

    if write_log:
        logger.info(f"Written")
    return


def read_json(file, write_log=False):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)

    if write_log:
        objects = len(data)
        logger.info(f"Read {objects:,} objects")
    return data


def write_json(file, data, indent=None, write_log=False):
    if write_log:
        objects = len(data)
        logger.info(f"Writing {objects:,} objects to {file}")

    with open(file, "w", encoding="utf8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    if write_log:
        logger.info(f"Written")
    return


class Config:
    def __init__(self, config_file):
        data = read_json(config_file)

        self.model = data["model"]
        self.static_dir = os.path.join(*data["static_dir"])
        self.state_dir = os.path.join(*data["state_dir"])
        self.output_dir = os.path.join(*data["output_dir"])

        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        return


class GPT:
    def __init__(self, model):
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.model_candidate_tokens = {
            "gpt-3.5-turbo": {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
            },
            "gpt-4": {
                "gpt-4": 8192,
                "gpt-4-32k": 32768,
            }
        }
        return

    def get_specific_tokens_model(self, text_in, out_tokens):
        in_token_list = self.tokenizer.encode(text_in)
        in_tokens = len(in_token_list)
        tokens = in_tokens + out_tokens

        for candidate, max_tokens in self.model_candidate_tokens.get(self.model, {}).items():
            if max_tokens >= tokens:
                break
        else:
            candidate = ""

        return in_tokens, candidate

    def run_gpt(self, text_in, out_tokens, retry_count=3, initial_delay=2, max_delay=10, timeout=30):
        """
        Runs the GPT model with improved error handling and timeout.

        :param text_in: The input text for the GPT model.
        :param out_tokens: Number of output tokens.
        :param retry_count: Number of retries on timeout. Default is 3.
        :param initial_delay: Initial delay in seconds before retrying. Doubles each retry, up to max_delay. Default is 2 seconds.
        :param max_delay: Maximum delay in seconds between retries. Default is 10 seconds.
        :param timeout: Timeout in seconds for each GPT request. Default is 30 seconds.
        """

        in_tokens, specific_tokens_model = self.get_specific_tokens_model(text_in, out_tokens)
        if not specific_tokens_model:
            return ""

        delay = initial_delay
        for attempt in range(retry_count):
            print("GPT running . . . ")
            try:
                completion = openai.ChatCompletion.create(
                    model=specific_tokens_model,
                    n=1,
                    messages=[{"role": "user", "content": text_in}],
                    request_timeout=timeout
                )
                return completion.choices[0].message.content
            except Timeout:
                logger.warning(f"Timeout occurred, retrying {attempt + 1}/{retry_count}...")
                print("Press Enter to retry or 'q' to quit...")
                user_input = input()
                if user_input.lower() == 'q':
                    break
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                return ""

        logger.error("Maximum retries reached. Unable to get a response.")
        return ""

class State:
    def __init__(self, save_file=""):
        self.save_file = save_file

        self.log = ""
        self.completed_stages = []
        self.current_stage = ""
        self.closed_ended = {}
        self.open_ended = {}
        self.adaptive = {}
        self.adaptive_sessions = 0
        self.persona_metrics = {"Introvert_Extrovert":0, "Sensing_iNtuition":0, "Thinking_Feeling":0, "Judging_Perceiving":0}
        self.mbtipersona = ""
        self.summary = ""
        return

    def save(self):
        data = {
            "log": self.log,
            "completed_stages": self.completed_stages,
            "current_stage": self.current_stage,
            "closed_ended": self.closed_ended,
            "open_ended": self.open_ended,
            "adaptive": self.adaptive,
            "persona_metrics": self.persona_metrics,
            "mbtipersona": self.mbtipersona,
            "summary": self.summary
        }
        write_json(self.save_file, data, indent=2)
        return

    def load(self):
        data = read_json(self.save_file)
        self.log = data["log"]
        self.completed_stages = data["completed_stages"]
        self.current_stage = data["current_stage"]
        self.closed_ended = data["closed_ended"]
        self.open_ended = data["open_ended"]
        self.adaptive = data["adaptive"]
        self.persona_metrics = data["persona_metrics"]
        self.mbtipersona = data["mbtipersona"]
        self.summary = data["summary"]
        return


class PersonaAdept:
    def __init__(self, config):
        self.static_dir = config.static_dir
        self.state_dir = config.state_dir
        self.output_dir = config.output_dir
        self.summary_file = ""
        self.gpt = GPT(config.model)
        # self.gpt4 = GPT("gpt-4")

        self.user_prompt_to_text = {}
        self.question_list_to_text = {}
        self.max_saves = 3
        self.state = State() # to save current state

        self.action_to_text = {
            "exit": "離開測驗",
            "start_closed_ended": "開始選擇題",
            "revise_closed_ended": "修改選擇題",
            "start_open_ended": "開始開放式問答",
            "revise_open_ended": "修改開放式問答",
            "start_adaptive": "開始追加問題",
            "cont_adaptive": "繼續追加問題",
            "end": "結束測驗"
        }

        # load prompt text
        user_prompt_dir = os.path.join(self.static_dir, "user_prompt")
        filename_list = os.listdir(user_prompt_dir)
        for filename in filename_list:
            user_prompt = filename[:-4]
            user_prompt_file = os.path.join(user_prompt_dir, filename)
            self.user_prompt_to_text[user_prompt] = read_txt(user_prompt_file)
        
        question_list_dir = os.path.join(self.static_dir, "question_list")
        filename_list = os.listdir(question_list_dir)
        for filename in filename_list:
            question_list = filename[:-4]
            question_list_file = os.path.join(question_list_dir, filename)
            self.question_list_to_text[question_list] = read_txt(question_list_file)

        # print("User prompt:", self.user_prompt_to_text)

        # load animal data
        # for animal in self.all_animal_list:
        #     animal_file = os.path.join(self.static_dir, "animal", f"{animal}.json")
        #     self.animal_data[animal] = read_json(animal_file)

        return

    def run_start(self):
        # get start type
        user_prompt = self.user_prompt_to_text["start"]
        while True:
            text_in = input(user_prompt)
            if text_in == "1":
                start_type = "new"
                break
            elif text_in == "2":
                start_type = "load"
                break

        # get save file usage
        save_list_text = "\n選擇檔案：\n"
        saveid_to_exist = {}
        for i in range(self.max_saves):
            save_id = str(i + 1)
            save_file = os.path.join(self.state_dir, f"save_{save_id}.json")
            if os.path.exists(save_file):
                saveid_to_exist[save_id] = True
                save_list_text += f"[{save_id}] 開啟舊檔\n"
            else:
                saveid_to_exist[save_id] = False
                save_list_text += f"[{save_id}] 開新檔案\n"

        # get save file
        user_prompt = f"{save_list_text}\n使用檔案： "
        while True:
            text_in = input(user_prompt)
            if start_type == "new":
                if text_in in saveid_to_exist:
                    use_save_id = text_in
                    break
            else:
                if saveid_to_exist.get(text_in, False):
                    use_save_id = text_in
                    break

        # initialize state
        self.summary_file = os.path.join(self.output_dir, f"summary_{use_save_id}.txt")
        use_save_file = os.path.join(self.state_dir, f"save_{use_save_id}.json")
        self.state = State(use_save_file)
        if start_type == "new":
            user_prompt = self.user_prompt_to_text["opening"]
            input(user_prompt + "\n開始人格測驗(按換行繼續)... ")
            self.state.log += user_prompt
            self.state.save()
        else:
            self.state.load()

        self.state.completed_stages.append("start")

        self.run_loop()
        return


    def run_loop(self):


        while True:

            action = self.get_action()

            if action == "exit":
                break

            elif action == "start_closed_ended":
                self.do_closed_ended()
                
            elif action == "revise_closed_ended":
                self.do_closed_ended()

            elif action == "start_open_ended":
                self.do_open_ended()

            elif action == "revise_open_ended":
                self.do_open_ended()

            elif action == "start_adaptive":
                self.do_adaptive()
            
            elif action == "cont_adaptive":
                self.do_adaptive()

            elif action == "end":
                self.do_end()

        return
    
    def get_action(self):
            # get available actions
            action_list = ["exit"]

            if "closed_ended" not in self.state.completed_stages:
                action_list.append("start_closed_ended")

            if "closed_ended" in self.state.completed_stages:
                action_list.append("revise_closed_ended")
                action_list.append("start_open_ended")

            if "open_ended" in self.state.completed_stages:
                action_list.append("revise_open_ended")
                action_list.append("start_adaptive")
                action_list.append("end")
            
            if "adaptive" in self.state.completed_stages:
                action_list.append("cont_adaptive")
                action_list.append("end")

            action_list_text = "\n執行：\n"
            actionid_to_action = {}
            for i, action in enumerate(action_list):
                action_id = str(i)
                actionid_to_action[action_id] = action
                action_text = self.action_to_text[action]
                action_list_text += f"({action_id}) {action_text}\n"

            # get action
            user_prompt = f"{action_list_text}\n你的下一步： "
            while True:
                text_in = input(user_prompt)
                if text_in in actionid_to_action:
                    use_action = actionid_to_action[text_in]
                    break

            return use_action


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

    def do_closed_ended(self):
        # print("start_closed_ended")

        user_prompt = self.user_prompt_to_text["closed_ended_start"]
        input(user_prompt + "\n... ")
        self.state.log += user_prompt

        # Introvert_Extrovert
        ans_1 = int(input("Q.1 在社交場合中，相較於獲得能量並感到精力充沛，我更傾向於感到疲倦和需要獨處來恢復精力："))
        ans_1 = 6 - ans_1

        # Sensing_iNtuition
        ans_2 = int(input("Q.2 在理解新概念或處理新信息時，相較於實際觀察和具體細節，我更傾向於運用抽象概念和整體理解："))

        # Thinking_Feeling
        ans_3 = int(input("Q.3 當需要做重要決策時，相較於依賴客觀的事實，我更傾向於考慮他人的感受和價值觀："))

        # Judging_Perceiving
        ans_4 = int(input("Q.4 處於工作或學習的壓力下，相較於提前計劃、組織，我更傾向於彈性應對、保持選項開放："))

        self.state.closed_ended = {
                "ans_1": ans_1,
                "ans_2": ans_2,
                "ans_3": ans_3,
                "ans_4": ans_4
            }
        
        self.state.log += f"\n\nQ.1 在社交場合中，相較於獲得能量並感到精力充沛，我更傾向於感到疲倦和需要獨處來恢復精力：{6-ans_1}\nQ.2 在理解新概念或處理新信息時，相較於實際觀察和具體細節，我更傾向於運用抽象概念和整體理解：{ans_2}\nQ.3 當需要做重要決策時，相較於依賴客觀的事實，我更傾向於考慮他人的感受和價值觀：{ans_3}\nQ.4 處於工作或學習的壓力下，相較於提前計劃、組織，我更傾向於彈性應對、保持選項開放：{ans_4}"
        
        # 1~5 -> -5~5 
        convert = lambda x: int(x*2.5 - 7.5)
        self.state.persona_metrics = {"Introvert_Extrovert":convert(ans_1), "Sensing_iNtuition":convert(ans_2), 
                                "Thinking_Feeling":convert(ans_3), "Judging_Perceiving":convert(ans_4)}
        
        self.state.completed_stages.append("closed_ended")
        self.state.current_stage = "open_ended"
        self.state.save()
        return
    
    def do_open_ended(self):
        # print("start_open_ended")

        user_prompt = self.user_prompt_to_text["open_ended_start"]
        input(user_prompt + "\n... ")
        self.state.log += user_prompt


        # Session 1
        ques = self.question_list_to_text["open_ended_ques"].split("/")[0]
        response = str(input(ques))
        gpt_in = self.question_list_to_text["scoring_prompt_first"].split("/")[0]
        gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", f"題目: '{ques}' \n使用者回覆: '{response}'")

        # print(gpt_in)
        
        score = self.gpt.run_gpt(gpt_in, 200)
        while not self.assert_format(score, 'score_-5_to_5', N=True):
            time.sleep(2)
            score = self.gpt.run_gpt(gpt_in, 200)
        
        followup = ""
        if score == 'N':
            followup = str(input("請說明仔細一點: "))
            gpt_in = self.question_list_to_text["scoring_prompt_followup"].split("/")[0]
            gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", f"題目: '{ques}' \n使用者回覆: '{response}'，並補充: '{followup}'")
            # print(gpt_in)

            score = self.gpt.run_gpt(gpt_in, 200)
            while not self.assert_format(score, 'score_-5_to_5'):
                time.sleep(2)
                score = self.gpt.run_gpt(gpt_in, 200)

        session_1_result = {"ques": ques, "response": response, "followup": followup, "score": int(score)}
        self.state.log += f"\n{ques}:{response}(補充:{followup}) 在-5(最接近Introvert)到5(最接近Extrovert) 獲得{score}分"
        self.state.open_ended.update({"session_1": session_1_result})
        self.state.persona_metrics["Introvert_Extrovert"] += int(score)
        self.state.save()


        # Session 2
        ques = self.question_list_to_text["open_ended_ques"].split("/")[1]
        response = str(input(ques))
        gpt_in = self.question_list_to_text["scoring_prompt_first"].split("/")[1]
        gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", f"題目: '{ques}' \n使用者回覆: '{response}'")

        # print(gpt_in)
        
        score = self.gpt.run_gpt(gpt_in, 200)
        while not self.assert_format(score, 'score_-5_to_5', N=True):
            time.sleep(2)
            score = self.gpt.run_gpt(gpt_in, 200)
        
        followup = ""
        if score == 'N':
            followup = str(input("請說明仔細一點: "))
            gpt_in = self.question_list_to_text["scoring_prompt_followup"].split("/")[1]
            gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", f"題目: '{ques}' \n使用者回覆: '{response}'，並補充: '{followup}'")
            # print(gpt_in)

            score = self.gpt.run_gpt(gpt_in, 200)
            while not self.assert_format(score, 'score_-5_to_5'):
                time.sleep(2)
                score = self.gpt.run_gpt(gpt_in, 200)

        session_2_result = {"ques": ques, "response": response, "followup": followup, "score": int(score)}
        self.state.open_ended.update({"session_2": session_2_result})
        self.state.log += f"\n{ques}:{response}(補充:{followup}) 在-5(最接近Sensing)到5(最接近iNtuition) 獲得{score}分"
        self.state.persona_metrics["Sensing_iNtuition"] += int(score)
        self.state.save()


        # Session 3
        ques = self.question_list_to_text["open_ended_ques"].split("/")[2]
        response = str(input(ques))
        gpt_in = self.question_list_to_text["scoring_prompt_first"].split("/")[2]
        gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", f"題目: '{ques}' \n使用者回覆: '{response}'")

        # print(gpt_in)
        
        score = self.gpt.run_gpt(gpt_in, 200)
        while not self.assert_format(score, 'score_-5_to_5', N=True):
            time.sleep(2)
            score = self.gpt.run_gpt(gpt_in, 200)
        
        followup = ""
        if score == 'N':
            followup = str(input("請說明仔細一點: "))
            gpt_in = self.question_list_to_text["scoring_prompt_followup"].split("/")[2]
            gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", f"題目: '{ques}' \n使用者回覆: '{response}'，並補充: '{followup}'")
            # print(gpt_in)

            score = self.gpt.run_gpt(gpt_in, 200)
            while not self.assert_format(score, 'score_-5_to_5'):
                time.sleep(2)
                score = self.gpt.run_gpt(gpt_in, 200)

        session_3_result = {"ques": ques, "response": response, "followup": followup, "score": int(score)}
        self.state.open_ended.update({"session_3": session_3_result})
        self.state.log += f"\n{ques}:{response}(補充:{followup}) 在-5(最接近Thinking)到5(最接近Feeling) 獲得{score}分"
        self.state.persona_metrics["Thinking_Feeling"] += int(score)
        self.state.save()


        # Session 4
        ques = self.question_list_to_text["open_ended_ques"].split("/")[3]
        response = str(input(ques))
        gpt_in = self.question_list_to_text["scoring_prompt_first"].split("/")[3]
        gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", f"題目: '{ques}' \n使用者回覆: '{response}'")

        # print(gpt_in)
        
        score = self.gpt.run_gpt(gpt_in, 200)
        while not self.assert_format(score, 'score_-5_to_5', N=True):
            time.sleep(2)
            score = self.gpt.run_gpt(gpt_in, 200)
        
        followup = ""
        if score == 'N':
            followup = str(input("請說明仔細一點: "))
            gpt_in = self.question_list_to_text["scoring_prompt_followup"].split("/")[3]
            gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", f"題目: '{ques}' \n使用者回覆: '{response}'，並補充: '{followup}'")
            # print(gpt_in)

            score = self.gpt.run_gpt(gpt_in, 200)
            while not self.assert_format(score, 'score_-5_to_5'):
                time.sleep(2)
                score = self.gpt.run_gpt(gpt_in, 200)

        session_4_result = {"ques": ques, "response": response, "followup": followup, "score": int(score)}
        self.state.open_ended.update({"session_4": session_4_result})
        self.state.log += f"\n{ques}:{response}(補充:{followup}) 在-5(最接近Judging)到5(最接近Perceiving) 獲得{score}分"
        self.state.persona_metrics["Judging_Perceiving"] += int(score)
        self.state.save()

        self.state.completed_stages.append("open_ended")
        self.state.current_stage = "adaptive"
        self.state.save()

        return
    
    def do_adaptive(self):
        # print("start_adaptive")

        # Calculate the accumulated score
        scores = self.state.persona_metrics
        # print(scores)

        if (scores["Introvert_Extrovert"]>10 or scores["Introvert_Extrovert"]<-10) and (scores["Sensing_iNtuition"]>10 or scores["Sensing_iNtuition"]<-10) and (scores["Thinking_Feeling"]>10 or scores["Thinking_Feeling"]<-10) and (scores["Judging_Perceiving"]>10 or scores["Judging_Perceiving"]<-10):
            # All persona metrics reach 10 or -10
            self.do_end()

        distances_to_goal = [min(scores["Introvert_Extrovert"]+10, 10-scores["Introvert_Extrovert"]), min(scores["Sensing_iNtuition"]+10, 10-scores["Sensing_iNtuition"]), min(scores["Thinking_Feeling"]+10, 10-scores["Thinking_Feeling"]), min(scores["Judging_Perceiving"]+10, 10-scores["Judging_Perceiving"])]
        # print(distances_to_goal)

        next_ques_category = distances_to_goal.index(max(distances_to_goal))
        # print(next_ques_category)

        ques_categories = {0:"adaptive_I_E", 1:"adaptive_S_N", 2:"adaptive_T_F", 3:"adaptive_J_P"}
        # print(self.question_list_to_text[ques_categories[next_ques_category]].split("/"))

        adaptive_ques = random.choice(self.question_list_to_text[ques_categories[next_ques_category]].split("/"))
        adaptive_response = str(input(adaptive_ques))
        
        gpt_in = self.question_list_to_text["scoring_prompt_first"].split("/")[next_ques_category]
        gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", f"題目: '{adaptive_ques}' \n使用者回覆: '{adaptive_response}'")
        # print(gpt_in)
        
        score = self.gpt.run_gpt(gpt_in, 200)
        while not self.assert_format(score, 'score_-5_to_5', N=True):
            time.sleep(2)
            score = self.gpt.run_gpt(gpt_in, 200)

        followup = ""
        if score == 'N':
            followup = str(input("請說明仔細一點: "))
            gpt_in = self.question_list_to_text["scoring_prompt_followup"].split("/")[next_ques_category]
            gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", f"題目: '{adaptive_ques}' \n使用者回覆: '{adaptive_response}'，並補充: '{followup}'")
            print(gpt_in)

            score = self.gpt.run_gpt(gpt_in, 200)
            while not self.assert_format(score, 'score_-5_to_5'):
                time.sleep(2)
                score = self.gpt.run_gpt(gpt_in, 200)

        adaptive_result = {"ques": adaptive_ques, "response": adaptive_response, "followup": followup, "score": int(score)}
        self.state.adaptive_sessions += 1
        self.state.adaptive["adaptive_"+str(self.state.adaptive_sessions)] = adaptive_result
        self.state.log += f"\n{adaptive_ques}:{adaptive_response}(補充:{followup}) 在{ques_categories[next_ques_category]}題目-5到5分 獲得{score}分"
        
        ques_metrics = {0:"Introvert_Extrovert", 1:"Sensing_iNtuition", 2:"Thinking_Feeling", 3:"Judging_Perceiving"}
        self.state.persona_metrics[ques_metrics[next_ques_category]] += int(score)
        self.state.save()

        if "adaptive" not in self.state.completed_stages:
            self.state.completed_stages.append("adaptive")
        self.state.save()
        return
        

    def do_end(self):

        # Summarize the user's responses
        all_text = re.sub(r"\n+", "\n", self.state.log).strip()
        all_text = all_text.replace("\n:", ' ')
        # print(all_text)


        pm = self.state.persona_metrics
        key_result = f"使用者在結束選擇題、開放式問答題、客製化問題後" \
                    f"在Introvert_Extrovert指標達到了{pm['Introvert_Extrovert']} (0為中立、靠近10為E、靠近-10為I)" \
                    f"在Sensing_iNtuition指標達到了{pm['Sensing_iNtuition']} (0為中立、靠近10為N、靠近-10為S)" \
                    f"在Thinking_Feeling指標達到了{pm['Thinking_Feeling']} (0為中立、靠近10為F、靠近-10為T)" \
                    f"在Judging_Perceiving指標達到了{pm['Judging_Perceiving']} (0為中立、靠近10為P、靠近-10為J)"
        gpt_in = str(self.question_list_to_text["key_result_prompt"])
        gpt_in = gpt_in.replace("INPUT_PLACEHOLDER", key_result)
        # print(gpt_in)

        mbtipersona = self.gpt.run_gpt(gpt_in, 200, timeout=60)
        while not self.assert_format(mbtipersona, 'MBTI'):
            time.sleep(5)
            mbtipersona = self.gpt.run_gpt(gpt_in, 200, timeout=60)

        print(self.user_prompt_to_text["end"] + str(mbtipersona))
        self.state.mbtipersona = mbtipersona
        self.state.save()

        gpt_in = self.question_list_to_text["summary_instruction"]
        gpt_in = gpt_in.replace("MBTI_PERSINALITY", str(mbtipersona))
        gpt_in = gpt_in.replace("ALL_LOG_TEXT", str(all_text))
        
        summary = self.gpt.run_gpt(gpt_in, 200, timeout=180)
        print(summary)
        self.state.summary = summary
        self.state.save()

        return

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="pa_config.json")
    arg = parser.parse_args()

    openai.api_key = input("OpenAI API Key: ")

    config = Config(arg.config_file)
    assess = PersonaAdept(config)
    assess.run_start()
    return


if __name__ == "__main__":
    
    main()
    sys.exit()
