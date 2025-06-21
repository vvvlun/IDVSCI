import logging
import re
import ollama
import torch.nn.functional
import numpy as np
import json
import os
from datetime import datetime
import random
import sys
sys.path.append('../agentscope-main/src')

from tqdm import tqdm
from agentscope.memory import TemporaryMemory
from agentscope.message import Msg
from utils.prompt import Prompts
from utils.scientist_utils import (
    extract_scientist_names,
    team_description,
    n2s,
    convert_you_to_other,
    team_description_detail,
    format_msg,
    formated_msg2str,
    read_txt_files_as_dict,
    extract_between_json_tags,
    extract_metrics,
    paper_search,
    strip_non_letters,
    save2database,
    count_team,
    top_three_indices,
    extract_first_number,
    most_frequent_element
)

class Team:
    def __init__(self, team_name, log_dir, info_dir):
        # attrs
        self.team_name = team_name
        self.state = 1
        self.epoch = -1
        self.teammate = []
        self.memory = TemporaryMemory(None)
        self.topic = None
        self.idea = None
        self.abstract = None
        self.citation_id = None
        self.self_review = None
        self.paper_review = None

        # state log
        self.state_log = {
            1: 'WAIT',
            2: 'TOPIC',
            3: 'IDEA',
            4: 'CHECK',
            5: 'ABSTRACT',
            6: 'REVIEW',
            7: 'FINISH'
        }

        # state action
        self.state_action = {
            2: self.select_topic,
            3: self.generate_idea,
            4: self.check_novelty,
            5: self.generate_abstract,
            6: self.generate_review
        }

        # init log file dir
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.info_file = f"{info_dir}/{current_time}_{self.team_name}_dialogue.json"
        self.log_file = f"{log_dir}/{current_time}_{self.team_name}_dialogue.log"

        # Check if log file exists and delete it
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        self.logger = logging.getLogger(self.team_name)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file)
        self.logger.addHandler(fh)

    # execute action based on team state
    def action_excution(self, platform):
        if self.state in self.state_log.keys():
            print(f'{"="*20} Epoch: {self.epoch} || BEGIN {self.state_log[self.state]} PROCESS {"="*20}')

        action = self.state_action.get(self.state, None)
        if action is not None:
            action(platform)
            print(f'{"="*20} Epoch: {self.epoch} || FINISH {self.state_log[self.state]} PROCESS {"="*20}')

    def _get_agent_topics(self, agent):
        """从agent对应的author文件中提取研究主题列表
        Args:
            agent: Agent实例
        Returns:
            list: 研究主题列表
        """
        try:
            # 从agent名称中提取数字
            agent_number = ''.join(filter(str.isdigit, agent.name))

            # 构建文件路径
            file_path = f"/mnt/data1/xchen/software/GCNDC-SCI/SciSci/Authors/books/books/author_{agent_number}.txt"

            # 读取文件内容
            with open(file_path, 'r') as f:
                content = f.read()

            # 使用正则表达式提取方括号中的topics
            topics_pattern = r"you have researched on following topics \[(.*?)\]"
            match = re.search(topics_pattern, content)

            if match:
                # 提取方括号中的内容并处理
                topics_str = match.group(1)
                # 分割字符串，去除引号和空格
                topics = [t.strip().strip("'") for t in topics_str.split(',')]
                return topics
            else:
                print(f"Warning: Could not find research topics in file for agent {agent.name}")
                return []

        except Exception as e:
            print(f"Warning: Could not get research topics for agent {agent.name}: {str(e)}")
            return []

    def _evaluate_topic_relevance(self, agent, current_topic):
        """评估agent的研究领域与当前主题的相关性
        Args:
            agent: Agent实例
            current_topic: str, 当前讨论的主题
        Returns:
            float: 相似度分数 (0-100的分数)
        """
        # 首先获取agent的研究主题
        agent_topics = self._get_agent_topics(agent)

        if not agent_topics:  # 如果没有找到研究主题
            print(f"Warning: No research topics found for agent {agent.name}")
            return 0.0

        # 获取当前主题的嵌入向量
        topic_embedding = ollama.embeddings(
            model="mxbai-embed-large",
            prompt=current_topic
        )['embedding']

        similarities = []
        for topic in agent_topics:
            # 获取每个研究领域的嵌入向量
            topic_vec = ollama.embeddings(
                model="mxbai-embed-large",
                prompt=topic
            )['embedding']

            # 计算余弦相似度
            sim = torch.nn.functional.cosine_similarity(
                torch.tensor(topic_embedding).unsqueeze(0),
                torch.tensor(topic_vec).unsqueeze(0),
                dim=-1
            )[0].item()

            # 将相似度转换为百分比分数(0-100)
            percentage = max(0, (sim + 1) * 50)  # 将[-1,1]映射到[0,100]
            similarities.append(percentage)

        # 返回最高的相似度分数
        return max(similarities) if similarities else 0.0

    # 记录agent与主题相关性
    def _log_team_expertise(self, platform, current_topic, stage):
        """记录团队成员与当前主题的相关性"""
        teammate = platform.id_to_agent(self.teammate)
        leader = teammate[0]
        other_members = teammate[1:]

        # 创建分析报告
        analysis_report = [
            f"\n{'=' * 20} Team Expertise Analysis {'=' * 20}",
            f"Stage: {stage}",
            f"Current Topic: {current_topic}",
            "\nTeam Members Relevance Analysis:"
        ]

        # 评估leader的相关性
        leader_relevance = self._evaluate_topic_relevance(
            leader,  # 直接传入agent实例
            current_topic,
        )

        leader_topics = self._get_agent_topics(leader)  # 获取topics用于显示
        leader_report = (
            f"Leader ({leader.name}):\n"
            f"- Research Topics: {', '.join(leader_topics)}\n"
            f"- Topic Relevance: {leader_relevance:.1f}% "
            f"({'High' if leader_relevance >= 70 else 'Medium' if leader_relevance >= 40 else 'Low'})"
        )
        analysis_report.append(leader_report)

        # 评估其他成员的相关性
        for member in other_members:
            member_relevance = self._evaluate_topic_relevance(
                member,  # 直接传入agent实例
                current_topic,
            )
            member_topics = self._get_agent_topics(member)  # 获取topics用于显示
            member_report = (
                f"\nMember ({member.name}):\n"
                f"- Research Topics: {', '.join(member_topics)}\n"
                f"- Topic Relevance: {member_relevance:.1f}% "
                f"({'High' if member_relevance >= 70 else 'Medium' if member_relevance >= 40 else 'Low'})"
            )
            analysis_report.append(member_report)

        analysis_report.append(f"\n{'=' * 60}")

        # 将完整报告作为一个字符串
        full_report = '\n'.join(analysis_report)

        # 同时打印到控制台和记录到日志
        print(full_report)
        self.log_dialogue('Team Expertise Analysis', full_report)

    # general group discussion process
    def group_discuss(self, platform, prompt :str = None):
        # prompt is used to start and guide the discussion
        # for each turn, in group_discuss, all dialogue history is stored in dialogue_history but not in agent memory
        # after finishing each discussion turn, agent1 will summarize dialogue_history and add a summarization into team_history

        # get team_history
        team_history = self.memory
        # get teammate
        teammate = platform.id_to_agent(self.teammate)
        # init dialogue_history
        dialogue_history = TemporaryMemory(None)
        # init exit state
        exit = False
        # output return dialogue history, summarization of the last turn, and memory of the last turn
        output = {}
        # start discussing
        if len(teammate)==1:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        else:
            group_max_discuss_iteration = platform.group_max_discuss_iteration

        for turn in range(group_max_discuss_iteration):
            said = []
            # init turn_memory for each turn
            turn_history = TemporaryMemory(None)
            agent_num = 0
            for agent in teammate:
                if agent.name in said:
                    continue
                else:
                    said.append(agent.name)

                agent_prompt = format_msg(
                    # current team
                    Msg(name="current team members", role="user", content=','.join(self.teammate)),
                    # team history
                    Msg(name="Summarizations of previous team discussions", role="user", content='')
                    if team_history.size()>0 else None,
                    team_history.get_memory(recent_n=platform.recent_n_team_mem_for_retrieve),
                    # prompt
                    Msg(name="user", role="user", content=prompt),
                    # dialogue history
                    Msg(name="Summarizations of previous turns in current team discussion", role="user", content='')
                    if dialogue_history.size()>0 else None,
                    dialogue_history.get_memory(recent_n=turn),
                    # turn history
                    Msg(name="Discussions in this turn", role="user", content='')
                    if turn_history.size()>0 else None,
                    turn_history.get_memory(recent_n=agent_num)
                )
                # add reply to turn_history
                reply = agent.prompt_reply(agent_prompt, add_memory = False, use_memory = False)
                if reply.content!=None and len(reply.content)>0:
                    self.log_dialogue(agent.name,reply.content)
                involved_scientist = extract_scientist_names(reply.content)
                print(involved_scientist)
                # judge whether someone is called to join the team
                for scientist_index in involved_scientist:
                    if scientist_index not in self.teammate:
                        if "by the way" in reply.content or "By the way" in reply.content:
                            hint = Msg(name=self.teammate[0],role="user",content=reply.content)
                            # invite new team member to comment
                            x = platform.id2agent[scientist_index].reply(hint, use_memory=False, use_RAG=False)
                            if x.content is not None:
                                said.append(scientist_index)
                                self.teammate.append(scientist_index)
                                self.log_dialogue(platform.id2agent[scientist_index].name, x.content)
                                teammate.append(platform.id2agent[scientist_index])

                turn_history.add(reply)
                agent_num = agent_num + 1
                # discussion is finished
                if 'exit' in reply:
                    exit = True
                    break

            # summarize this turn's discussion
            history = format_msg(
                # team history
                Msg(name="Summarizations of previous team discussions", role="user", content='')
                if team_history.size()>0 else None,
                team_history.get_memory(recent_n=platform.recent_n_team_mem_for_retrieve),
                # dialogue history
                Msg(name="Summarizations of previous turns in current team discussion", role="user", content='')
                if dialogue_history.size()>0 else None,
                dialogue_history.get_memory(recent_n=turn),
            )
            x = teammate[0].summarize(history = history, content = turn_history.get_memory(recent_n=agent_num))
            self.log_dialogue(teammate[0].name, x.content)
            turn_summarization = Msg(name="summarizations of turn{}".format(turn+1), role="user",
                                     content=x.content)

            if exit or turn==group_max_discuss_iteration-1:
                output['last_turn_summarization'] = turn_summarization
                output['last_turn_history'] = turn_history
                break
            else:
                dialogue_history.add(turn_summarization)

        output['dialogue_history'] = dialogue_history
        self.teammate = platform.agent_to_id(teammate)
        return output

    def select_topic(self, platform):
        # prompt to start discussing select_topic
        discuss_result = self.group_discuss(platform, Prompts.to_start_topic_discussion)
        team_history = self.memory
        dialogue_history = discuss_result['dialogue_history']
        last_turn_history = discuss_result['last_turn_history']
        last_turn_summarization = discuss_result['last_turn_summarization']

        answer_prompt = format_msg(
            # team history
            Msg(name="Summarizations of previous team discussions", role="user", content='')
            if team_history.size()>0 else None,
            team_history.get_memory(recent_n=platform.recent_n_team_mem_for_retrieve),
            # dialogue history
            Msg(name="Summarizations of previous turns in current team discussion", role="user", content='')
            if dialogue_history.size()>0 else None,
            dialogue_history.get_memory(recent_n=platform.group_max_discuss_iteration),
            # turn history
            Msg(name="Discussions in this turn", role="user", content='')
            if last_turn_history.size()>0 else None,
            last_turn_history.get_memory(recent_n=last_turn_history.size()),
            # answer_prompt
            Msg(name="user", role="user", content=Prompts.to_ask_if_ready_give_topic)
        )
        answer = platform.id2agent[self.teammate[0]].prompt_reply(answer_prompt, add_memory = False, use_memory=False)
        self.log_dialogue('user', platform.id2agent[self.teammate[0]].model.format(answer_prompt))
        self.log_dialogue(platform.id2agent[self.teammate[0]].name, answer.content)
        answer_pattern = re.compile(r'action\s*1', re.IGNORECASE)

        # update dialogue history
        dialogue_history.add(last_turn_summarization)
        dialogue_history.add(answer)

        # check whether agent is ready to answer
        if answer_pattern.search(answer.content) or team_history.size()>=1:
            self.state = 3
            history_prompt = format_msg(
                # team history
                Msg(name="Summarizations of previous team discussions", role="user", content='')
                if team_history.size()>0 else None,
                team_history.get_memory(recent_n=platform.recent_n_team_mem_for_retrieve),
                # dialogue history
                Msg(name="Summarizations of previous turns in current team discussion", role="user", content='')
                if dialogue_history.size()>0 else None,
                dialogue_history.get_memory(recent_n=platform.group_max_discuss_iteration),
                # turn history
                Msg(name="Discussions in this turn", role="user", content='')
                if last_turn_history.size()>0 else None,
                last_turn_history.get_memory(recent_n=last_turn_history.size()),
            )
            topic_prompt = format_msg(
                # answer,
                # topic_prompt
                Msg(name="user", role="user", content=Prompts.to_ask_topic.replace("[history_prompt]", formated_msg2str(history_prompt)))
            )
            topic = platform.id2agent[self.teammate[0]].prompt_reply(topic_prompt, add_memory = False)
            self.log_dialogue(self.teammate[0],topic.content)
            self.topic = extract_between_json_tags(topic.content,num=1)
            self.topic = strip_non_letters(self.topic.split("Topic")[1])

            # update dialogue history
            dialogue_history.add(topic)

        # update team_history
        history = format_msg(
            # team history
            Msg(name="Summarizations of previous team discussions", role="user", content='')
            if team_history.size()>0 else None,
            team_history.get_memory(recent_n=platform.recent_n_team_mem_for_retrieve)
        )
        team_history.add(Msg(name="summarizations of one topic discussion", role="user",
                             content=platform.id2agent[self.teammate[0]].summarize(history = history, content = dialogue_history.get_memory(recent_n=platform.group_max_discuss_iteration))))
        self.memory = team_history

   # review方式：agent按照顺序，根据对话历史对idea进行修改
   #  def generate_idea(self, platform):
   #      topic = self.topic
   #      old_idea = None
   #      best_idea = None
   #      idea_list = []
   #      mark_list = []
   #      # search related paper about the topic
   #      selected_topics = strip_non_letters(topic.split("Selected Topics:")[-1])
   #      paper_reference, cite_paper = platform.reference_paper(selected_topics, platform.cite_number)
   #
   #      teammate = platform.id_to_agent(self.teammate)
   #      idea_judge = True
   #
   #      if len(teammate)==1:
   #          group_max_discuss_iteration = platform.group_max_discuss_iteration
   #      else:
   #          group_max_discuss_iteration = platform.group_max_discuss_iteration
   #
   #      for turn in range(group_max_discuss_iteration):
   #          # discuss the idea
   #          for agent in teammate:
   #              idea_prompt = Prompts.prompt_task+Prompts.prompt_existing_idea.format(old_idea)+ \
   #                            Prompts.prompt_topic.format(selected_topics)+Prompts.prompt_reference.format(paper_reference)+ \
   #                            Prompts.prompt_response
   #              agent_prompt = format_msg(
   #                  # prompt
   #                  Msg(name="user", role="user", content=idea_prompt),
   #              )
   #              # 应该是在这里改
   #              reply = agent.prompt_reply(agent_prompt, add_memory = False, use_memory = False, use_RAG=False)
   #              self.log_dialogue('user',idea_prompt)
   #              self.log_dialogue(agent.name,reply.content)
   #              old_idea = extract_between_json_tags(reply.content, num=1)
   #              if "Title" in old_idea:
   #                  idea_key = old_idea.split("Title")[1]
   #                  idea_key = strip_non_letters(idea_key.split("Experiment")[0])
   #              else:
   #                  idea_key = old_idea.split("Idea")[1]
   #                  idea_key = strip_non_letters(idea_key.split("Experiment")[0])
   #              paper_reference, cite_paper_new = platform.reference_paper(idea_key, platform.cite_number)
   #              cite_paper = list(set(cite_paper).union(cite_paper_new))
   #
   #              # find the metric
   #              split_keywords = ['Clarity', 'Feasibility', 'Novelty']
   #              metrics = extract_metrics(old_idea, split_keywords)
   #              if best_idea != None:
   #                  if old_idea == best_idea:
   #                      idea_judge=True
   #                      print("exit early!!!!!!")
   #                      break
   #                  best_metrics = extract_metrics(best_idea, split_keywords)
   #                  old_count = 0
   #                  best_count = 0
   #                  for split_keywork in split_keywords:
   #                      if metrics[split_keyword]==None:
   #                          break
   #                      if split_keyword=='Novelty':
   #                          old_count = old_count + 2*metrics[split_keyword]
   #                      else:
   #                          old_count = old_count + metrics[split_keyword]
   #                      if best_metrics[split_keyword]==None:
   #                          break
   #                      best_count = best_count + best_metrics[split_keyword]
   #                  if old_count>=best_count:
   #                      best_idea = old_idea
   #                      idea_list.append(old_idea)
   #                      mark_list.append(old_count)
   #              else:
   #                  idea_list.append(old_idea)
   #                  best_idea = old_idea
   #              # if all metrics are larger than 8, then over
   #              for split_keyword in split_keywords:
   #                  if metrics[split_keyword]==None:
   #                      break
   #                  if metrics[split_keyword]<11:
   #                      idea_judge=False
   #                      break
   #              if idea_judge:
   #                  best_idea=old_idea
   #                  break
   #          if idea_judge:
   #              break
   #      if self.idea == None:
   #          if len(idea_list)>3:
   #              indices = top_three_indices(mark_list)
   #              idea_list = [idea_list[i] for i in indices]
   #              self.idea = idea_list
   #          else:
   #              self.idea = idea_list
   #      print("Candidate Idea:")
   #      print(self.idea)
   #      if platform.skip_check:
   #          self.state=5
   #      else:
   #          self.state=4
   #      self.citation_id = cite_paper
   #      print(len(self.citation_id))

#   review方式：除leader外agent修改后，返回结果给leader统计
    def generate_idea(self, platform):
        # Initialize variables
        topic = self.topic
        selected_topics = strip_non_letters(topic.split("Selected Topics:")[-1])
        paper_reference, cite_paper = platform.reference_paper(selected_topics, platform.cite_number)

        num_agents = len(self.teammate)
        idea_list = []
        mark_list = []
        teammate = platform.id_to_agent(self.teammate)
        # 设置讨论轮次
        if len(teammate) == 1:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        else:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        # 每轮讨论
        for turn in range(group_max_discuss_iteration):
            # 特殊情况：只有一个agent（leader）
            if num_agents == 1:
                idea_prompt = (Prompts.prompt_task +
                               Prompts.prompt_topic.format(selected_topics) +
                               Prompts.prompt_reference.format(paper_reference) +
                               """Please ensure your response follows this format:
                                    {
                                        "Title": "Your idea title",
                                        "Idea": "Detailed description of your idea",
                                        "Experiment": "Experimental design"
                                    }
                                    """ +
                               Prompts.prompt_response)

                initial_idea = teammate[0].prompt_reply(
                    format_msg(Msg(name="user", role="user", content=idea_prompt)),
                    add_memory=False, use_memory=False, use_RAG=False
                )
                self.log_dialogue('user', idea_prompt)
                self.log_dialogue(teammate[0].name, initial_idea.content)

                current_idea = extract_between_json_tags(initial_idea.content, num=1)
                if current_idea:
                    # 自我评估
                    eval_prompt = self._create_eval_prompt(current_idea)
                    eval_response = teammate[0].prompt_reply(
                        format_msg(Msg(name="user", role="user", content=eval_prompt)),
                        add_memory=False, use_memory=False
                    )
                    metrics = extract_metrics(eval_response.content, ['Clarity', 'Feasibility', 'Novelty'])
                    if metrics:
                        idea_score = (metrics.get('Clarity', 5) * 1.0 +
                                      metrics.get('Feasibility', 5) * 1.0 +
                                      metrics.get('Novelty', 5) * 2.0) / 4.0
                        idea_list.append(current_idea)
                        mark_list.append(idea_score)

            else:
                # 在检查新颖性前评估团队专业度
                self._log_team_expertise(platform, selected_topics, "Idea Generation")
                # 每个agent都生成一个想法
                for current_generator in range(num_agents):
                    # Generate initial idea
                    idea_prompt = (Prompts.prompt_task +
                                   Prompts.prompt_topic.format(selected_topics) +
                                   Prompts.prompt_reference.format(paper_reference) +
                                   """Please ensure your response follows this format:
                                      {
                                          "Title": "Your idea title",
                                          "Idea": "Detailed description of your idea",
                                          "Experiment": "Experimental design"
                                      }
                                      """ +
                                   Prompts.prompt_response)

                    initial_idea = teammate[current_generator].prompt_reply(
                        format_msg(Msg(name="user", role="user", content=idea_prompt)),
                        add_memory=False, use_memory=False, use_RAG=False
                    )
                    self.log_dialogue('user', idea_prompt)
                    self.log_dialogue(teammate[current_generator].name, initial_idea.content)

                    current_idea = extract_between_json_tags(initial_idea.content, num=1)

                    # Update paper references
                    if current_idea:
                        if "Title" in current_idea:
                            idea_key = current_idea.split("Title")[1]
                            idea_key = strip_non_letters(idea_key.split("Experiment")[0])
                        else:
                            idea_key = current_idea.split("Idea")[1]
                            idea_key = strip_non_letters(idea_key.split("Experiment")[0])

                        paper_reference, cite_paper_new = platform.reference_paper(idea_key, platform.cite_number)
                        cite_paper = list(set(cite_paper).union(cite_paper_new))

                        # Collect modifications from other agents (除了生成者和leader)
                        modifications = []
                        for i in range(num_agents):
                            if i != current_generator and i != 0:
                                modify_prompt = f"""Please review and modify this idea. Focus on improving its clarity, feasibility, and novelty:
                                        Original idea:
                                        {current_idea}

                                        Please provide your modifications in JSON format."""

                                modification = teammate[i].prompt_reply(
                                    format_msg(Msg(name="user", role="user", content=modify_prompt)),
                                    add_memory=False, use_memory=False
                                )
                                self.log_dialogue(teammate[i].name, modification.content)
                                mod_idea = extract_between_json_tags(modification.content, num=1)
                                if mod_idea:
                                    modifications.append(mod_idea)

                        # Leader synthesizes modifications
                        if modifications:
                            synthesis_prompt = f"""As the team leader, please synthesize these modifications into a coherent improvement:
                                    Original idea: {current_idea}
                                    Team modifications: {json.dumps(modifications, indent=2)}

                                    Please provide a synthesized version that combines the best elements of all suggestions."""

                            leader_synthesis = teammate[0].prompt_reply(
                                format_msg(Msg(name="user", role="user", content=synthesis_prompt)),
                                add_memory=False, use_memory=False
                            )
                            self.log_dialogue(teammate[0].name, leader_synthesis.content)

                            # Original agent reflects on leader's synthesis
                            reflect_prompt = f"""Please reflect on the feedback and provide your final improved version:
                                    Your original idea: {current_idea}
                                    Team's synthesis (by leader): {leader_synthesis.content}

                                    Please provide your final improved version in JSON format, incorporating the valuable feedback while maintaining your original vision."""

                            final_reflection = teammate[current_generator].prompt_reply(
                                format_msg(Msg(name="user", role="user", content=reflect_prompt)),
                                add_memory=False, use_memory=False
                            )
                            self.log_dialogue(teammate[current_generator].name, final_reflection.content)
                            improved_idea = extract_between_json_tags(final_reflection.content, num=1)
                        else:
                            improved_idea = current_idea

                        # Evaluate the improved idea
                        if improved_idea:
                            # 评估改进后的想法
                            eval_prompt = self._create_eval_prompt(improved_idea)
                            eval_response = teammate[current_generator].prompt_reply(
                                format_msg(Msg(name="user", role="user", content=eval_prompt)),
                                add_memory=False, use_memory=False
                            )
                            metrics = extract_metrics(eval_response.content, ['Clarity', 'Feasibility', 'Novelty'])
                            idea_score = (metrics.get('Clarity', 5) * 1.0 +
                                          metrics.get('Feasibility', 5) * 1.0 +
                                          metrics.get('Novelty', 5) * 2.0) / 4.0
                            # 将改进后的想法和分数添加到总列表中
                            idea_list.append(improved_idea)
                            mark_list.append(idea_score)

                            idea_key = self.extract_idea_key(improved_idea)
                            paper_reference, cite_paper_new = platform.reference_paper(idea_key, platform.cite_number)
                            cite_paper = list(set(cite_paper).union(cite_paper_new))

        # 处理最终结果
        if self.idea is None:
            if len(idea_list) > 3:
                indices = top_three_indices(mark_list)
                self.idea = [idea_list[i] for i in indices]
            else:
                self.idea = idea_list

        print("Candidate Ideas:")
        print(self.idea)
        print(f"Total number of ideas generated: {len(idea_list)}")

        self.citation_id = cite_paper
        print(f"Number of citations: {len(self.citation_id)}")

        if platform.skip_check:
            self.state = 5
        else:
            self.state = 4

    # 更安全的方式来提取 idea_key
    def extract_idea_key(self, idea_text):
        try:
            if "Title" in idea_text:
                # 尝试通过 Title 提取
                idea_key = idea_text.split("Title")[1]
                idea_key = strip_non_letters(idea_key.split("Experiment")[0] if "Experiment" in idea_key else idea_key)
            elif "Idea" in idea_text:
                # 尝试通过 Idea 提取
                idea_key = idea_text.split("Idea")[1]
                idea_key = strip_non_letters(idea_key.split("Experiment")[0] if "Experiment" in idea_key else idea_key)
            else:
                # 如果都没有找到，使用整个文本
                idea_key = strip_non_letters(idea_text)
            return idea_key
        except Exception as e:
            print(f"[DEBUG] Error extracting idea key: {e}")
            print(f"[DEBUG] Original idea text: {idea_text}")
            return idea_text  # 返回原始文本作为后备方案

    def _create_eval_prompt(self, idea):
        return f"""Please evaluate this idea based on the following criteria:
                    1. Clarity (1-10): How clear and well-structured is the idea?
                       Score must be between 1 and 10, where 10 is the highest possible score.
                    2. Feasibility (1-10): How practical and implementable is the idea? For this evaluation reference, please carefully refer to relevant articles similar to this idea from 2011 and later, and the more articles with more than 4 citations, the higher the score will be assigned.
                       Score must be between 1 and 10, where 10 is the highest possible score.
                    3. Novelty (1-10): How innovative and original is the idea? For this evaluation reference, please carefully refer to related articles that are similar to this IDEA, with lower scores assigned to articles that are more similar to those before 2011, and higher scores assigned to articles that are more similar to those in 2011 and later.
                       Score must be between 1 and 10, where 10 is the highest possible score.

                    Note: Scores MUST NOT exceed 10 under any circumstances.

                    IMPORTANT RULES:
                    - All scores MUST be whole numbers (integers)
                    - Scores MUST be between 1 and 10
                    - NO decimal points allowed
                    - Example score: 7 (correct) | 7.5 (incorrect)

                    Idea to evaluate:
                    {idea}

                    Please provide your evaluation in JSON format:
                    {{
                        "Clarity": score (1-10),
                        "Feasibility": score (1-10),
                        "Novelty": score (1-10)
                    }}
                    Example response:
                    {{
                    "Clarity": 8,
                    "Feasibility": 7,
                    "Novelty": 9
                    }}"""

    #   review方式：邀请外部团队
    # def generate_idea(self, platform):
    #     topic = self.topic
    #     old_idea = None
    #     idea_list = []
    #     mark_list = []
    #
    #     # 获取相关论文
    #     selected_topics = strip_non_letters(topic.split("Selected Topics:")[-1])
    #     paper_reference, cite_paper = platform.reference_paper(selected_topics, platform.cite_number)
    #
    #     teammate = platform.id_to_agent(self.teammate)
    #     idea_judge = True
    #     group_max_discuss_iteration = platform.group_max_discuss_iteration
    #
    #     # 原团队生成ideas
    #     for turn in range(group_max_discuss_iteration):
    #         for agent in teammate:
    #             # 构建prompt
    #             idea_prompt = (Prompts.prompt_task +
    #                          Prompts.prompt_existing_idea.format(old_idea) +
    #                          Prompts.prompt_topic.format(selected_topics) +
    #                          Prompts.prompt_reference.format(paper_reference) +
    #                          Prompts.prompt_response)
    #
    #             agent_prompt = format_msg(
    #                 Msg(name="user", role="user", content=idea_prompt),
    #             )
    #
    #             # 获取回复
    #             reply = agent.prompt_reply(agent_prompt, add_memory=False, use_memory=False, use_RAG=False)
    #             self.log_dialogue('user', idea_prompt)
    #             self.log_dialogue(agent.name, reply.content)
    #
    #             # 提取idea
    #             old_idea = extract_between_json_tags(reply.content, num=1)
    #
    #             # 提取关键词用于搜索相关论文
    #             if "Title" in old_idea:
    #                 idea_key = old_idea.split("Title")[1]
    #                 idea_key = strip_non_letters(idea_key.split("Experiment")[0])
    #             else:
    #                 idea_key = old_idea.split("Idea")[1]
    #                 idea_key = strip_non_letters(idea_key.split("Experiment")[0])
    #
    #             # 更新参考文献
    #             paper_reference, cite_paper_new = platform.reference_paper(idea_key, platform.cite_number)
    #             cite_paper = list(set(cite_paper).union(cite_paper_new))
    #
    #             idea_list.append(old_idea)
    #
    #         # 获取外部评审团队
    #         review_team = platform.invite_external_review(
    #             team=self,
    #             topic=self.topic,
    #             idea=idea_list,
    #             selection_mode="similar"  # 可以根据需要修改选择模式
    #         )
    #
    #         lead_reviewer = review_team[0]
    #         other_reviewers = review_team[1:]
    #
    #         # 收集评审意见
    #         review_opinions = []
    #         for reviewer in other_reviewers:
    #             review_prompt = (
    #                     f"As an external reviewer, please evaluate the following research ideas based on the provided references:\n\n" +
    #                     "\n\n".join([f"Idea {i + 1}:\n{idea}" for i, idea in enumerate(idea_list)]) +
    #                     f"\n\nRelevant References:\n{paper_reference}\n\n" +
    #                     "Please evaluate each idea considering:\n" +
    #                     "1. Clarity (1-10): Is the idea clearly articulated?\n" +
    #                     "2. Feasibility (1-10): Can it be implemented with current technology?\n" +
    #                     "3. Novelty (1-10): How innovative compared to existing work?\n" +
    #                     "4. For the novelty, please carefully refer to Relevant References that are similar to this IDEA, with lower scores assigned to articles that are more similar to those before 2011, and higher scores assigned to articles that are more similar to those in 2011 and later.\n"+
    #                     "For the potential impact, please carefully refer to Relevant References similar to this idea from 2011 and later, and the more articles with more than 4 citations, the higher the score will be assigned.\n\n" +
    #                     "Format your response in JSON with ratings and comments for each idea."
    #             )
    #
    #             review = reviewer.prompt_reply(format_msg(
    #                 Msg(name="user", role="user", content=review_prompt)
    #             ))
    #             review_opinions.append(review.content)
    #             self.log_dialogue(reviewer.name, review.content)
    #
    #             # Leader决策
    #         leader_prompt = (
    #                 "As the lead reviewer, analyze these ideas and review opinions:\n\n" +
    #                 "\n\n".join([f"Reviewer {i + 1}:\n{opinion}" for i, opinion in enumerate(review_opinions)]) +
    #                 f"\n\nRelevant References:\n{paper_reference}\n\n" +
    #                 "Please select the top 3 ideas from the list based on reviewer ratings and comments.\n" +
    #                 "Provide your decision in JSON format:\n" +
    #                 "{\n" +
    #                 '  "selected_ideas": [\n' +
    #                 '    {"rank": 1, "idea_index": X, "score": 9.5, "reason": "..."}, // X should be 0-based index from idea_list\n' +
    #                 '    {"rank": 2, "idea_index": Y, "score": 8.7, "reason": "..."},\n' +
    #                 '    {"rank": 3, "idea_index": Z, "score": 8.2, "reason": "..."}\n' +
    #                 '  ]\n' +
    #                 "}"
    #         )
    #
    #         leader_decision = lead_reviewer.prompt_reply(format_msg(
    #             Msg(name="user", role="user", content=leader_prompt)
    #         ))
    #         self.log_dialogue(lead_reviewer.name, leader_decision.content)
    #
    #         # 直接从idea_list中提取选中的ideas
    #         try:
    #             # 尝试解析leader的决定
    #             json_str = extract_between_json_tags(leader_decision.content)
    #             if not json_str:
    #                 # 如果没有找到有效的JSON，使用备选格式
    #                 decision_json = {
    #                     "selected_ideas": [
    #                         {
    #                             "rank": 1,
    #                             "idea_index": 0,
    #                             "reason": "Default selection due to parsing error"
    #                         }
    #                     ]
    #                 }
    #                 print("Warning: Could not parse leader decision JSON, using default selection")
    #             else:
    #                 decision_json = json.loads(json_str)
    #
    #             # 从idea_list中提取选中的ideas
    #             selected_ideas = [idea_list[item["idea_index"]] for item in decision_json["selected_ideas"]]
    #
    #         except Exception as e:
    #             print(f"Error processing leader decision: {e}")
    #             # 发生错误时使用第一个idea作为默认选择
    #             selected_ideas = [idea_list[0]] if idea_list else []
    #
    #         # 更新团队状态
    #         self.idea = selected_ideas
    #         self.citation_id = cite_paper
    #         self.state = 5 if platform.skip_check else 4
    #
    #         print("Selected Ideas:")
    #         print(self.idea)
    #         print(f"Number of citations: {len(self.citation_id)}")

  # review方式：agent依次生成意见，最后统计结果
  #   def check_novelty(self, platform):
  #       existing_idea = self.idea
  #       idea_choices = ""
  #       for idea_index in range(len(existing_idea)):
  #           idea = existing_idea[idea_index]
  #           idea_choices = idea_choices+"Idea "+str(idea_index)+":\n"+idea+"\n"
  #       related_papers = []
  #       for idea_index in existing_idea:
  #           title = idea_index.split("Title")[1]
  #           title = strip_non_letters(title.split("Experiment")[0])
  #           if len(existing_idea)==3:
  #               cite_number = 3
  #           else:
  #               cite_number = 5
  #           _, related_paper = platform.reference_paper(title, cite_number)
  #
  #           related_papers = list(set(related_papers).union(related_paper))
  #
  #       paper_reference = ""
  #       for id in range(len(related_papers)):
  #           paper_index = related_papers[id]
  #           paper_reference = paper_reference+"Paper {}:".format(id+1)+"\n"
  #           paper_reference = paper_reference+"Title: "+platform.paper_dicts[paper_index]['title']+"\n"
  #           paper_reference = paper_reference+"Abstract: "+platform.paper_dicts[paper_index]['abstract']+"}"+"\n"
  #
  #       teammate = platform.id_to_agent(self.teammate)
  #       choice_list = []
  #       if len(teammate)==1:
  #           group_max_discuss_iteration = platform.group_max_discuss_iteration
  #       else:
  #           group_max_discuss_iteration = platform.group_max_discuss_iteration
  #       for turn in range(group_max_discuss_iteration):
  #           # discuss the idea
  #           for agent in teammate:
  #               # 测试用更短的上下文
  #               # idea_novelty_prompt = Prompts.prompt_idea_check + Prompts.prompt_idea_check_response.replace("{existing_idea}", idea_choices)
  #               idea_novelty_prompt = Prompts.prompt_idea_check + Prompts.prompt_idea_check_response.replace("{existing_idea}", idea_choices).replace("{last_query_results}",paper_reference)
  #               agent_prompt = format_msg(
  #                   # prompt
  #                   Msg(name="user", role="user", content=idea_novelty_prompt),
  #               )
  #               reply = agent.prompt_reply(agent_prompt, add_memory = False, use_memory = False, use_RAG=False)
  #               self.log_dialogue('user',idea_novelty_prompt)
  #               self.log_dialogue(agent.name,reply.content)
  #               old_idea = extract_between_json_tags(reply.content, num=1)
  #               idea_choice = extract_first_number(old_idea)
  #               if idea_choice == None:
  #                   idea_choice = 0
  #               choice_list.append(int(idea_choice))
  #
  #       final_choice = most_frequent_element(choice_list)
  #       if final_choice<0 or final_choice>=len(existing_idea):
  #           final_choice = len(existing_idea)-1
  #       try:
  #           self.idea = existing_idea[final_choice]
  #       except:
  #           self.idea = existing_idea[0]
  #       print("Final Idea:")
  #       print(self.idea)
  #       self.state=5


#   review方式：除leader外，其余agent返回新颖性意见。最后投票
    def check_novelty(self, platform):
        # 准备现有想法
        existing_idea = self.idea
        idea_choices = ""
        for idea_index in range(len(existing_idea)):
            idea_choices = idea_choices + "Idea " + str(idea_index) + ":\n" + existing_idea[idea_index] + "\n"

        # 收集相关论文
        related_papers = []
        for idea_index in existing_idea:
            if "Title" in idea_index:
                title = idea_index.split("Title")[1]
                title = strip_non_letters(title.split("Experiment")[0])
                cite_number = 3 if len(existing_idea) == 3 else 5
                _, related_paper = platform.reference_paper(title, cite_number)
                related_papers = list(set(related_papers).union(related_paper))
            else:
                print(f"Warning: 'Title' not found in idea_index: {idea_index}")
                continue

        # 获取团队成员
        teammate = platform.id_to_agent(self.teammate)
        leader = teammate[0]  # Scientist0 作为 leader
        other_members = teammate[1:] if len(teammate) > 1 else []

        # 为每个成员分配论文
        member_papers = self._distribute_papers(related_papers, len(other_members))

        # 第一轮：成员各自分析分配到的论文
        member_analyses = []
        for i, member in enumerate(other_members):
            paper_reference = self._format_paper_reference(member_papers[i], platform)
            analysis_prompt = f"""Based on these papers: {paper_reference}
            Please analyze how these papers relate to the following ideas: {idea_choices}
            Focus on novelty, potential impact, and technical feasibility.
            For the novelty, please carefully refer to related articles that are similar to this IDEA, with lower scores assigned to articles that are more similar to those before 2011, and higher scores assigned to articles that are more similar to those in 2011 and later.
            For the potential impact, please carefully refer to relevant articles similar to this idea from 2011 and later, and the more articles with more than 4 citations, the higher the score will be assigned.
            Provide a brief analysis for each idea."""

            response = member.prompt_reply(
                format_msg(Msg(name="user", role="user", content=analysis_prompt)),
                add_memory=False, use_memory=False, use_RAG=False
            )
            member_analyses.append(response.content)
            self.log_dialogue('user', analysis_prompt)
            self.log_dialogue(member.name, response.content)

        # 第二轮：leader 总结分析
        if member_analyses:
            summary_prompt = f"""As the team leader, review these analyses from your team members:
            {' '.join(member_analyses)}

            Synthesize their findings and provide a comprehensive evaluation of each idea in {idea_choices}
            Focus on novelty, feasibility, and potential impact.
            For the novelty, please carefully refer to related articles that are similar to this IDEA, with lower scores assigned to articles that are more similar to those before 2011, and higher scores assigned to articles that are more similar to those in 2011 and later.
            For the potential impact, please carefully refer to relevant articles similar to this idea from 2011 and later, and the more articles with more than 4 citations, the higher the score will be assigned."""

            leader_summary = leader.prompt_reply(
                format_msg(Msg(name="user", role="user", content=summary_prompt)),
                add_memory=False, use_memory=False, use_RAG=False
            )
            self.log_dialogue('user', summary_prompt)
            self.log_dialogue(leader.name, leader_summary.content)
        else:
            leader_summary = None

        # 第三轮：加权 Borda 计数投票
        votes = []
        for member in teammate:
            voting_prompt = f"""Based on {"the leader's summary:" + leader_summary.content if leader_summary else 'your analysis'},
                rank the ideas from best to worst (0 being best).
                Consider novelty, feasibility, and impact.
                For the novelty, please carefully refer to related articles that are similar to this IDEA, with lower scores assigned to articles that are more similar to those before 2011, and higher scores assigned to articles that are more similar to those in 2011 and later.
                For the potential impact, please carefully refer to relevant articles similar to this idea from 2011 and later, and the more articles with more than 4 citations, the higher the score will be assigned.
                {idea_choices}

                IMPORTANT: You must use the idea's numeric index (starting from 0) in your ranking.
                For example, if there are 3 ideas, your ranking should only contain numbers: 0, 1, or 2.

                Respond in JSON format:
                {{
                    "ranking": [list of idea indices as integers (0, 1, 2, etc.)],
                    "confidence_scores": [corresponding confidence scores from 1-10]
                }}

                Example response format:
                {{
                    "ranking": [0, 2, 1],
                    "confidence_scores": [9, 8, 7]
                }}"""

            vote_response = member.prompt_reply(
                format_msg(Msg(name="user", role="user", content=voting_prompt)),
                add_memory=False, use_memory=False, use_RAG=False
            )
            try:
                vote_data = json.loads(vote_response.content)
                votes.append(vote_data)
            except:
                # 如果解析失败，使用默认投票
                votes.append(
                    {"ranking": list(range(len(existing_idea))), "confidence_scores": [5] * len(existing_idea)})

        # 计算最终结果
        final_choice = self._weighted_borda_count(votes, len(existing_idea))

        try:
            self.idea = existing_idea[final_choice]
        except:
            self.idea = existing_idea[0]

        print("Final Idea:")
        print(self.idea)
        self.state = 5
#
    def _distribute_papers(self, papers, num_members):
        """将论文分配给团队成员"""
        if num_members == 0:
            return []

        # 确保每个成员至少获得一篇论文
        base_papers_per_member = len(papers) // num_members
        extra_papers = len(papers) % num_members

        distribution = []
        start_idx = 0
        for i in range(num_members):
            papers_for_member = base_papers_per_member + (1 if i < extra_papers else 0)
            end_idx = start_idx + papers_for_member
            distribution.append(papers[start_idx:end_idx])
            start_idx = end_idx

        return distribution

    def _format_paper_reference(self, paper_indices, platform):
        """格式化论文引用"""
        paper_reference = ""
        for idx, paper_index in enumerate(paper_indices):
            paper_reference += f"Paper {idx + 1}:\n"
            paper_reference += f"Title: {platform.paper_dicts[paper_index]['title']}\n"
            paper_reference += f"Abstract: {platform.paper_dicts[paper_index]['abstract']}\n"
        return paper_reference

    def _weighted_borda_count(self, votes, num_ideas):
        """实现加权 Borda 计数投票"""
        scores = [0] * num_ideas

        for vote in votes:
            ranking = vote["ranking"]
            confidence_scores = vote["confidence_scores"]

            for rank, (idea_idx, confidence) in enumerate(zip(ranking, confidence_scores)):
                try:
                    # 确保 idea_idx 是整数
                    idea_idx = int(idea_idx)
                    if 0 <= idea_idx < num_ideas:  # 验证索引范围
                        # Borda 分数 = (排名权重) * (信心分数)
                        # 排名权重 = 总选项数 - 当前排名
                        borda_score = (num_ideas - rank) * (confidence / 10.0)
                        scores[idea_idx] += borda_score
                except (ValueError, TypeError):
                    print(f"警告：无效的想法索引 {idea_idx}，已跳过")
                    continue

        # 如果所有分数都是0，返回0作为默认选择
        if all(score == 0 for score in scores):
            return 0

        return scores.index(max(scores))

  # review方式：邀请外审团队
  #   def check_novelty(self, platform):
  #       existing_idea = self.idea
  #       idea_choices = ""
  #       for idea_index in range(len(existing_idea)):
  #           idea = existing_idea[idea_index]
  #           idea_choices = idea_choices+f"Idea{idea_index}:\n{idea}\n"
  #
  #       # 获取相关论文
  #       related_papers=[]
  #       title = idea.split("Title")[1]
  #       title = strip_non_letters(title.split("Experiment")[0])
  #       cite_number = 3 if len(existing_idea) == 3 else 5
  #       _, related_paper = platform.reference_paper(title, cite_number)
  #       related_papers = list(set(related_papers).union(related_paper))
  #
  #       paper_reference = ""
  #       for id in range(len(related_papers)):
  #           paper_index = related_papers[id]
  #           paper_reference = paper_reference + "Paper {}:".format(id + 1) + "\n"
  #           paper_reference = paper_reference + "Title: " + platform.paper_dicts[paper_index]['title'] + "\n"
  #           paper_reference = paper_reference + "Abstract: " + platform.paper_dicts[paper_index]['abstract'] + "}" + "\n"
  #
  #       # 使用platform的invite_external_review获取外审团队
  #       review_team = platform.invite_external_review(
  #           team=self,
  #           topic=self.topic,
  #           idea=idea_choices,
  #           selection_mode="similar"  # 选择相关领域的专家
  #       )
  #
  #       choice_list = []
  #       # 每位评审给出评估
  #       for reviewer in review_team:
  #           novelty_prompt = (
  #               "As an external reviewer, evaluate these research ideas and select the most novel one:\n"
  #               f"Consider novelty, feasibility, and impact.\n"
  #               f"For the novelty, please carefully refer to related articles that are similar to this IDEA, with lower scores assigned to articles that are more similar to those before 2011, and higher scores assigned to articles that are more similar to those in 2011 and later."
  #               f"For the potential impact, please carefully refer to relevant articles similar to this idea from 2011 and later, and the more articles with more than 4 citations, the higher the score will be assigned."
  #               f"{idea_choices}\n"
  #               f"Reference papers:\n{paper_reference}\n"
  #               "Please provide your evaluation in JSON format:\n"
  #               "{\n"
  #               '  "selected_idea": X,  // Index number of most novel idea\n'
  #               '  "reasoning": "Your explanation for this choice..."\n'
  #               "}"
  #           )
  #
  #       agent_prompt = format_msg(
  #           Msg(name="user", role="user", content=novelty_prompt)
  #       )
  #       reply = reviewer.prompt_reply(agent_prompt, add_memory=False, use_memory=False, use_RAG=False)
  #       self.log_dialogue('user', novelty_prompt)
  #
  #       review_json = extract_between_json_tags(reply.content, num=1)
  #       idea_choice = extract_first_number(review_json)
  #       if idea_choice == None:
  #           idea_choice = 0
  #       choice_list.append(int(idea_choice))
  #
  #       # 多数投票决策
  #       final_choice = most_frequent_element(choice_list)
  #       if final_choice < 0 or final_choice >= len(existing_idea):
  #           final_choice = len(existing_idea) - 1
  #
  #       try:
  #           self.idea = existing_idea[final_choice]
  #       except:
  #           self.idea = existing_idea[0]
  #       print("Final Idea:")
  #       print(self.idea)
  #       self.state = 5
  # 计算整个索引库向量太大了，分批次做search
  #   def batch_search(self, index, query_vector, k, batch_size=2000):
  #       """分批执行搜索
  #       Args:
  #           index: FAISS索引
  #           query_vector: 查询向量
  #           k: 需要返回的总向量数
  #           batch_size: 每批处理的大小
  #       Returns:
  #           tuple: (距离数组, 索引数组)
  #       """
  #       n_batches = (k + batch_size - 1) // batch_size
  #       all_distances = []
  #       all_indices = []
  #
  #       for i in range(n_batches):
  #           current_k = min(batch_size, k - i * batch_size)
  #           if current_k <= 0:
  #               break
  #           D, I = index.search(query_vector, current_k)
  #           all_distances.append(D)
  #           all_indices.append(I)
  #
  #       # 合并结果
  #       final_distances = np.concatenate(all_distances, axis=1)
  #       final_indices = np.concatenate(all_indices, axis=1)
  #
  #       return final_distances, final_indices

    def generate_abstract(self, platform):
        idea = self.idea
        old_abstract = self.abstract
        teammate = platform.id_to_agent(self.teammate)

        if len(teammate)==1:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        else:
            group_max_discuss_iteration = platform.group_max_discuss_iteration

        for turn in range(group_max_discuss_iteration):
            # discuss the abstract
            for agent in teammate:
                if old_abstract == None:
                    abstract_prompt = Prompts.prompt_abstract+"\n"+idea+ \
                                      "\n"+Prompts.prompt_abstract_requirement+"\n"+Prompts.prompt_abstract_response
                else:
                    # the paper is not reviewed by reviewer
                    if self.paper_review == None:
                        # the paper is not reviewer by the team member
                        if self.self_review == None:
                            prompt_abstract_judgement = Prompts.prompt_abstract_judgement.replace("[Insert abstract here]",old_abstract)
                            abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response
                        else:
                            prompt_abstract_judgement = Prompts.prompt_abstract_judgement_self.replace("[Insert abstract here]",old_abstract)
                            prompt_abstract_judgement = prompt_abstract_judgement.replace("[Insert self_review comments]", self.self_review)
                            abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response
                    else:
                        prompt_abstract_judgement = Prompts.prompt_abstract_judgement_after_review.replace("[Insert Reviewer comments]",self.paper_review)
                        prompt_abstract_judgement = prompt_abstract_judgement.replace("[Insert abstract here]",old_abstract)
                        abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response
                agent_prompt = format_msg(
                    # prompt
                    Msg(name="user", role="user", content=abstract_prompt),
                )
                reply = agent.prompt_reply(agent_prompt, add_memory = False, use_memory = False, use_RAG=False)
                self.log_dialogue(agent.name, reply.content)
                old_abstract = extract_between_json_tags(reply.content, num=1)
                if old_abstract == None:
                    old_abstract = reply.content

        related_papers = []
        # 1. 处理摘要并生成查询向量
        def clean_abstract(text):
            """清理文本,移除标题和Abstract关键词"""
            # 如果文本包含"Title:"，移除标题部分
            if "Title:" in text:
                text = text.split("Title:")[1]
                # 找到第一个换行符,删除标题行
                text = text.split("\n", 1)[1]

            # 如果包含"Abstract",移除这个关键词
            if "Abstract:" in text:
                text = text.split("Abstract:")[1]
            elif "Abstract" in text:
                text = text.split("Abstract")[1]

            # 原有的清理步骤
            text = text.replace('\n', ' ')
            text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        Abstract = clean_abstract(old_abstract)
        # 添加文本预处理
        query_vector = ollama.embeddings(model="mxbai-embed-large", prompt=Abstract)
        query_vector = np.array([query_vector['embedding']])

        D_future, I_future = platform.gpu_future_index.search(query_vector, int(platform.cite_number/2))
        D, I = platform.gpu_index.search(query_vector, int(platform.cite_number/2))

        D_future = np.sqrt(D_future)
        D = np.sqrt(D)

        # 4. 收集相关论文
        # 处理未来论文
        for id in range(len(I_future[0])):
            paper_title = platform.paper_future_dicts[I_future[0][id]]['title']
            paper_abstract = platform.paper_future_dicts[I_future[0][id]]['abstract']
            paper_year = platform.paper_future_dicts[I_future[0][id]]['year']
            paper_citation = platform.paper_future_dicts[I_future[0][id]]['citation']
            paper_index = {}
            paper_index['title'] = paper_title
            paper_index['abstract'] = paper_abstract
            paper_index['year'] = paper_year
            paper_index['citation'] = paper_citation
            related_papers.append(paper_index)
        # 处理历史论文
        for id in range(len(I[0])):
            paper_title = platform.paper_dicts[I[0][id]]['title']
            paper_abstract = platform.paper_dicts[I[0][id]]['abstract']
            paper_year = platform.paper_dicts[I[0][id]]['year']
            paper_citation = platform.paper_dicts[I[0][id]]['citation']
            paper_index = {}
            paper_index['title'] = paper_title
            paper_index['abstract'] = paper_abstract
            paper_index['year'] = paper_year
            paper_index['citation'] = paper_citation
            related_papers.append(paper_index)

        # 5. 计算数据库平均引用次数和相对引用率
        total_citations = sum(int(paper['citation']) for paper in platform.paper_future_dicts
                              if str(paper['citation']).isdigit())
        total_papers = len(platform.paper_future_dicts)
        avg_citation = total_citations / total_papers
        self.log_dialogue('average_citation_in_database', str(avg_citation))

        # 按语义相似度排序并计算top 5的相对引用率
        # 将距离和论文索引组合在一起
        future_pairs = list(zip(D_future[0], range(len(I_future[0]))))

        # 合并并按距离排序
        all_papers_with_distance = []
        for dist, idx in future_pairs:
            paper = platform.paper_future_dicts[I_future[0][idx]]
            all_papers_with_distance.append((dist, paper))

        # 按距离升序排序（距离越小越相似）
        sorted_papers = [paper for _, paper in sorted(all_papers_with_distance, key=lambda x: x[0])]
        top_5_papers = sorted_papers[:min(5, len(sorted_papers))]
        relative_citations = [int(paper['citation']) if isinstance(paper['citation'], str) else paper['citation']
                              for paper in top_5_papers]
        avg_relative_citation = sum(relative_citations) / len(relative_citations) if relative_citations else 0
        self.log_dialogue('average_relative_citation_of_top_5', str(avg_relative_citation))

        # 6. eval with embedding similarity
        abs = []
        our_abs = strip_non_letters(old_abstract.split('Abstract')[1])
        abs.append(ollama.embeddings(model="mxbai-embed-large", prompt=our_abs)['embedding'])
        for paper_id in range(int(platform.cite_number/2), len(related_papers)):
            related_astract = related_papers[paper_id]['abstract']
            abs.append(ollama.embeddings(model="mxbai-embed-large", prompt=related_astract)['embedding'])

        sim = []
        for emb_id in range(1, len(abs)):
            sim.append(torch.nn.functional.cosine_similarity(torch.tensor(abs[0]).unsqueeze(0),
                                                             torch.tensor(abs[emb_id]).unsqueeze(0), dim=-1)[0].item())
        self.log_dialogue('embedding similarity', str(sim))

        # eval with LLM
        print('related papers:')
        print(len(related_papers))
        if len(related_papers)>0:
            self.log_dialogue('arxiv',related_papers)
        # find paper successfully
        if len(related_papers)>0:
            abstract_check_prompt = Prompts.prompt_abstract_check.replace("[Insert your abstract here]", old_abstract)
            cite_abstract = ""
            # word = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
            word = ['A', 'B', 'C', 'D']
            split_keywords = []
            for paper_id in range(int(platform.cite_number/2), len(related_papers)):
                cite_abstract = cite_abstract + str(paper_id-int(platform.cite_number/2)+1) + ". Abstract {}: ".format(word[paper_id-int(platform.cite_number/2)]) + "Title: " + related_papers[paper_id]['title'] + "\n" + "Abstract: " + related_papers[paper_id]['abstract'] + "\n"
                split_keywords.append('Written Abstract vs {}'.format(word[paper_id-int(platform.cite_number/2)]))
            abstract_check_prompt = abstract_check_prompt.replace("[Insert ref abstract here]", cite_abstract)
            abstract_check_prompt = abstract_check_prompt + "\n" + Prompts.prompt_response_check
            
            agent_prompt = format_msg(
                # prompt
                Msg(name="user", role="user", content=abstract_check_prompt),
            )
            reply = teammate[0].prompt_reply(agent_prompt, add_memory = False, use_memory = False, use_RAG=False)
            self.log_dialogue(teammate[0].name, reply.content)
            print("abstract_check:")
            print(split_keywords)
            comparison = extract_between_json_tags(reply.content)
            metric = extract_metrics(comparison, split_keywords=split_keywords)
            abstract_use = True
            for split_keyword in split_keywords:
                if metric[split_keyword]>=70:
                    abstract_use = False
                    self.abstract = old_abstract
                    break
            self.abstract = old_abstract
            print('Final Abstract:')
            print(self.abstract)
            # stop early
            self.state=7

            # do not stop early

            # if abstract_use:
            #     team.state=6
            #     team.self_review=None
            # # if the abstract is too similar one time, go to revise, otherwise back to generate idea
            # else:
            #     if team.self_review!=None:
            #         team.state=3
            #         team.idea = None
            #         team.abstract = None
            #         team.citation_id = None
            #         team.self_review = None
            #         team.paper_review = None
            #     else:
            #         team.self_review = reply.content

        else:
            print('Check Fail!!!!!!')
            if self.abstract == None:
                self.abstract = old_abstract
                print('Final Abstract:')
                print(self.abstract)
                self.state=6

    def generate_review(self, platform):
        # paper reviewer by reviewer
        print('current reviewing paper from {}'.format(self.teammate))
        old_abstract = self.abstract
        prompt = Prompts.prompt_review_require_simple.replace("{paper}", old_abstract)
        mark_sum = 0
        self.paper_review==None
        for _ in range(platform.reviewer_num):
            agent_prompt = format_msg(
                # prompt
                Msg(name="user", role="user", content=prompt),
            )
            reply = platform.reviewer_pool[_].prompt_reply(agent_prompt, add_memory = False, use_memory = False, use_RAG=False)
            self.log_dialogue(platform.reviewer_pool[_].name, reply.content)
            split_keywords = ['Overall']
            metric = extract_metrics(reply.content, split_keywords)
            if self.paper_review == None:
                self.paper_review = platform.reviewer_pool[_].name+":\n"+reply.content
            else:
                self.paper_review = self.paper_review+"\n"+platform.reviewer_pool[_].name+":\n"+reply.content
            for split_keyword in split_keywords:
                if metric[split_keyword] == None:
                    mark_sum = mark_sum + platform.default_mark
                else:
                    mark_sum = mark_sum + metric[split_keyword]
        if mark_sum>=(5*platform.reviewer_num):
            print('paper accept!!!!!!')
            self.state=platform.over_state
            title = old_abstract.split("Abstract")[0]
            title = strip_non_letters(title.split("Title")[1])
            abstract = strip_non_letters(old_abstract.split("Abstract")[1])
            file_dict={}
            file_dict['title']=title
            file_dict['abstract']=abstract
            file_dict['id'] = len(platform.paper_dicts)
            file_dict['authors'] = self.teammate
            file_dict['cite_papers'] = self.citation_id
            platform.paper_dicts.append(file_dict)
            # add embedding into list
            embedding_list = []
            response = ollama.embeddings(model="mxbai-embed-large", prompt=abstract)
            embedding_list.append(response["embedding"])
            response = np.array(embedding_list)
            platform.gpu_index.add(response)
        else:
            self.state = 5

    def log_dialogue(self, name, content):
        self.logger.info(f'Epoch:{self.epoch} | {self.state_log[self.state]} | {name}:{content}')
        self.logger.info(f'{"="*40}')

    def save_team_info(self):
        team_info = {
            'teammate':self.teammate,
            'topic':self.topic,
            'idea':self.idea,
            'abstract':self.abstract
        }
        print(f'{"="*20} SAVE TEAM INFO {"="*20}')
        with open(self.info_file, 'w') as json_file:
            json.dump(team_info, json_file, indent=4)

if __name__=='__main__':
    team1 = Team('LPL')
    team2 = Team('LCK')
    team1.log_dialogue('sam', 'LPL win!')
    team2.log_dialogue('tom', 'LCK win!')
    team1.log_dialogue('sam', 'LPL win again !')
    team2.log_dialogue('tom', 'LCK win again !')
