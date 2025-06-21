import sys
import os
import numpy as np
import json
import re
import ollama
from functools import partial
import faiss
import random

sys.path.append('../agentscope-main/src')
import agentscope
from agentscope.rag import KnowledgeBank
from agentscope.agents import SciAgent
from agentscope.message import Msg

from sci_team.SciTeam import Team
from utils.prompt import Prompts
from utils.scientist_utils import (
    team_description,
    convert_you_to_other,
    team_description_detail,
    read_txt_files_as_dict,
    extract_between_json_tags,
    count_team,
)

class Platform:
    r"""Platform."""

    def __init__(self,
                 model_configuration: str = '/home/yuwl/GCNDC-SCI/sci_platform/configs/model_configs.json',
                 agent_num: int = 1,
                 paper_folder_path: str = "/home/yuwl/Social_Science-main/SciSci/Papers/Papers/papers",
                 future_paper_folder_path: str = "/home/yuwl/Social_Science-main/SciSci/Papers/Papers_Future/papers_future",
                 author_info_dir: str = '/home/yuwl/Social_Science-main/SciSci/Authors/books/books',
                 adjacency_matrix_dir: str = '/home/yuwl/Social_Science-main/SciSci/adj',
                 agent_model_config_name: str = 'ollama_qwq',
                 review_model_config_name: str = 'ollama_llama3.1_70b',
                 # agent_model_config_name: str = 'ollama_qwen2.5',
                 # review_model_config_name: str = 'gpt-4o-2024-08-06',  # reviewer用最新的GPT-4o
                 knowledgeBank_config_dir: str = "/home/yuwl/Social_Science-main/sci_platform/configs/knowledge_config.json",
                 log_dir: str = 'logs',
                 info_dir: str = "team_info",
                 hop_num: int = 2,
                 group_max_discuss_iteration: int = 7, # 6， 7
                 recent_n_team_mem_for_retrieve: int = 1,
                 team_limit: int = 2,
                 check_iter: int = 5,
                 review_num: int = 2,
                 max_teammember: int = 9, # 3， 7， 9
                 cite_number: int = 8,
                 default_mark: int = 4,
                 skip_check: bool = False,
                 over_state: int = 8,
                 begin_state: int = 1
                 ):
        self.agent_num = agent_num
        self.paper_folder_path = paper_folder_path
        self.paper_future_folder_path = future_paper_folder_path
        self.author_info_dir = author_info_dir
        self.adjacency_matrix_dir = adjacency_matrix_dir
        self.group_max_discuss_iteration = group_max_discuss_iteration
        self.recent_n_team_mem_for_retrieve = recent_n_team_mem_for_retrieve
        # how many teams for one agent is allowed
        self.team_limit = team_limit
        # how many times to try paper search
        self.check_iter = check_iter
        # the number of reviewer
        self.reviewer_num = review_num
        # the max team member in a team
        self.max_teammember = max_teammember
        # cite how many paper when generating the idea
        self.cite_number = cite_number
        # default review mark
        self.default_mark = default_mark
        # check novelty
        self.skip_check = skip_check
        # current state for the over of team activity
        self.over_state = over_state
        # current state for the begin of team activity
        self.begin_state = begin_state
        # output dir
        self.log_dir = log_dir
        self.info_dir = info_dir

        # for quality, the team of one member will think more times
        self.think_times = max_teammember+1

        # load k-hop adjacency matrix
        self.degree_int2word = ['one', 'two', 'three', 'four', 'five']
        # self.adjacency_matrix = np.loadtxt(
        #     '{}/{}-hop_adj_matrix.txt'.format(self.adjacency_matrix_dir, self.degree_int2word[hop_num-1]), dtype=int)
        self.adjacency_matrix = np.loadtxt(
            '{}/adjacency.txt'.format(self.adjacency_matrix_dir), dtype=int)

        # check if agent_num is valid
        if self.agent_num is None:
            self.agent_num = len(self.adjacency_matrix)
        else:
            assert self.agent_num <= len(self.adjacency_matrix)

        # init agentscope
        agentscope.init(model_configs=model_configuration)

        # init knowledge bank
        if knowledgeBank_config_dir is not None:
            self.knowledge_bank = self.init_knowledgeBank(knowledgeBank_config_dir)

        # init agent pool
        self.agent_pool = [self.init_agent(str(agent_id), agent_model_config_name, '{}/author_{}.txt'.format(self.author_info_dir, agent_id)) for agent_id in range(len(self.adjacency_matrix))]
        # 若想随机
        # self.agent_pool = [
        #     self.init_agent(
        #         str(random.randint(1, 100)),  # 随机生成1到100之间的整数
        #         agent_model_config_name,
        #         '{}/author_{}.txt'.format(self.author_info_dir, agent_id)  # 保持文件路径不变
        #     )
        #     for agent_id in range(len(self.adjacency_matrix))
        # ]
        self.reviewer_pool = [self.init_reviewer(str(agent_id), review_model_config_name) for agent_id in range(self.reviewer_num)]
        self.id2agent = {}
        for agent in self.agent_pool:
            self.knowledge_bank.equip(agent, agent.knowledge_id_list)
            self.id2agent[agent.name] = agent
        # team pool
        self.team_pool = []
        # ===正确的===
        agent_id = 1
        for agent in self.agent_pool[:self.agent_num]:
            team_agent = []
            team_index = []
            team_index.append(agent.name)
            team_dic = Team(team_name = str(agent_id)+','+str(1),
                            log_dir = self.log_dir,
                            info_dir = self.info_dir)
            team_dic.teammate = team_index
            team_agent.append(team_dic)
            self.team_pool.append(team_agent)
            agent_id = agent_id + 1
        # ===测试背景一样时agent表现===
        # for agent in self.agent_pool[:self.agent_num]:
        #     team_agent = []
        #     team_index = [agent.name] * self.max_teammember  # 所有成员都是当前科学家，例如 ["Scientist0", "Scientist0", "Scientist0"]
        #     team_dic = Team(
        #         team_name=f"{agent.name}_team",
        #         log_dir=self.log_dir,
        #         info_dir=self.info_dir
        #     )
        #     team_dic.teammate = team_index
        #     team_agent.append(team_dic)
        #     self.team_pool.append(team_agent)

        # init hint
        self.HostMsg = partial(Msg, name="user", role="user", echo=True)

        # paper embedding list
        cpu_index = faiss.read_index("/home/yuwl/Social_Science-main/SciSci/Embeddings/faiss_index.index")
        res = faiss.StandardGpuResources()  
        self.gpu_index = faiss.index_cpu_to_gpu(res, 1, cpu_index)

        cpu_future_index = faiss.read_index("/home/yuwl/Social_Science-main/SciSci/Embeddings/faiss_index_future.index")
        future_res = faiss.StandardGpuResources()  
        self.gpu_future_index = faiss.index_cpu_to_gpu(future_res, 3, cpu_future_index)  # 将索引移到 GPU

        self.paper_dicts = read_txt_files_as_dict(self.paper_folder_path)
        self.paper_future_dicts = read_txt_files_as_dict(self.paper_future_folder_path)

# 新添加，用于邀请外部团队
    def invite_external_review(self, team, topic, idea, selection_mode="random"):
        # 1. 计算topic的embedding
        topic_vector = np.array([ollama.embeddings(model="mxbai-embed-large", prompt=topic)['embedding']])

        # 2. 选择lead reviewer
        available_agents = [a for a in self.agent_pool if a not in team.teammate]
        if selection_mode == "random":
            lead_reviewer = np.random.choice(available_agents)

        else:
            # 计算所有可用agent与topic的相似度
            similarities = []
            for agent in available_agents:
                # 获取agent的研究主题并合并成字符串
                agent_topics = team._get_agent_topics(agent)
                agent_topics_text = ", ".join(agent_topics)

                agent_vector = np.array([ollama.embeddings(
                    model="mxbai-embed-large",
                    prompt=agent_topics_text)['embedding']])

                # 修改相似度计算方法
                # 1. 计算余弦相似度
                dot_product = np.dot(topic_vector.flatten(), agent_vector.flatten())
                topic_norm = np.linalg.norm(topic_vector)
                agent_norm = np.linalg.norm(agent_vector)
                cosine_similarity = dot_product / (topic_norm * agent_norm)

                # 2. 将相似度归一化到0-1范围
                normalized_similarity = (cosine_similarity + 1) / 2

                similarities.append((agent, normalized_similarity))

            # 根据模式选择lead reviewer
            if selection_mode == "similar":
                similarities.sort(key=lambda x: x[1], reverse=True)  # 相似度从高到低
            else:  # dissimilar
                similarities.sort(key=lambda x: x[1])  # 相似度从低到高

            lead_reviewer = similarities[0][0]

            # 记录相似度到日志
            team.log_dialogue(
                "system",
                f"Lead reviewer selection ({selection_mode} mode):\n" +
                "\n".join([f"{agent.name}: {score:.4f}" for agent, score in similarities])
            )

        # 3-4. 选择其他成员的代码保持不变，但使用相同的相似度计算方法
        review_team_size = np.random.choice([4, 8])
        remaining_count = review_team_size - 1
        review_team = [lead_reviewer]
        remaining_agents = [a for a in available_agents if a != lead_reviewer]

        if selection_mode == "random":
            selected_agents = list(np.random.choice(remaining_agents, remaining_count, replace=False))
        else:
            similarities = []
            for agent in remaining_agents:
                agent_vector = np.array([ollama.embeddings(
                    model="mxbai-embed-large",
                    prompt=agent.sys_prompt)['embedding']])

                # 使用相同的相似度计算方法
                dot_product = np.dot(topic_vector.flatten(), agent_vector.flatten())
                topic_norm = np.linalg.norm(topic_vector)
                agent_norm = np.linalg.norm(agent_vector)
                cosine_similarity = dot_product / (topic_norm * agent_norm)
                normalized_similarity = (cosine_similarity + 1) / 2

                similarities.append((agent, normalized_similarity))

            if selection_mode == "similar":
                similarities.sort(key=lambda x: x[1], reverse=True)
            else:  # dissimilar
                similarities.sort(key=lambda x: x[1])

            selected_agents = [pair[0] for pair in similarities[:remaining_count]]

            team.log_dialogue(
                "system",
                f"Team member selection ({selection_mode} mode):\n" +
                "\n".join([f"{agent.name}: {score:.4f}" for agent, score in similarities])
            )

        review_team.extend(selected_agents)

        # 5. 记录最终团队组成
        team.log_dialogue(
            "system",
            f"External review team formed:\nLead reviewer: {lead_reviewer.name}\n" +
            f"Team members: {[agent.name for agent in selected_agents]}"
        )

        return review_team

    def init_reviewer(self, agent_id, agent_model_config_name):
        agent = SciAgent(
            name='Paper Reviewer{}'.format(agent_id),
            model_config_name=agent_model_config_name,
            sys_prompt=Prompts.prompt_review_system,
        )
        return agent

    def init_agent(self, agent_id, agent_model_config_name, information_path):
        # load author info
        with open(information_path, 'r') as file:
            prompt = file.read()

        agent = SciAgent(
            name='Scientist{}'.format(agent_id),
            model_config_name=agent_model_config_name,
            sys_prompt=prompt,
            knowledge_id_list = ["author_information"],
            similarity_top_k=2,
            log_retrieval=False,
            recent_n_mem_for_retrieve=2,
        )

        return agent

    def init_knowledgeBank(self, knowledgeBank_config_dir):
        knowledge_bank = KnowledgeBank(configs=knowledgeBank_config_dir)

        # alternatively, we can easily input the configs to add data to RAG
        knowledge_bank.add_data_as_knowledge(
            knowledge_id="author_information",
            emb_model_name="ollama_embedding-mxbai-embed-large",
            data_dirs_and_types={
                self.author_info_dir: [".txt"],
            },
        )
        return knowledge_bank

    # ===正确的===
    def select_coauthors(self,):
        team_list = self.team_pool
        scientists = self.agent_pool[:self.agent_num]
        # decide whether the scientist wants to find partners
        for agent_index in range(len(scientists)):
            # avoid too many teams
            if count_team(team_list[agent_index], self.over_state)>=self.team_limit:
                # choose to enforcely add the owner as a team
                if team_list[agent_index][0].state==1:
                    team_list[agent_index][0].state=2
                continue
            scientists[agent_index].sys_prompt = scientists[agent_index].sys_prompt + Prompts.role
            hint = self.HostMsg(content=Prompts.ask_choice.format_map(
                {
                    "Scientist_name": scientists[agent_index].name,
                    "All_team": team_description(team_list[agent_index], self.over_state)
                },
            ),
            )
            x = scientists[agent_index].reply(hint)
            team_list[agent_index][0].log_dialogue('user',hint.content)
            team_list[agent_index][0].log_dialogue(scientists[agent_index].name,x.content)
            match = re.search(r'action\s*(\d+)', extract_between_json_tags(x.content,num=1), re.IGNORECASE)

            if match is not None:
                # 当 action2，代理选择独立行动
                if int(match.group(1)) == 2:
                    print("Single Agent Independently!")
                    team_list[agent_index][0].state = 2
                    continue
            else:
                print("没有找到匹配项，x.content:", x.content)

            # when action2, the agent choose to act independently
            # if int(match.group(1))==2:
            #     print("Single Agent Independently!")
            #     team_list[agent_index][0].state=2
            #     continue

            # use prompts to select scientists
            scientist = scientists[agent_index].name
            name = int(scientist[9:])

            arr = self.adjacency_matrix[name,:]
            arr[agent_index] = 0
            probabilities = arr / np.sum(arr)

            selected_indices = np.random.choice(len(arr), size=self.max_teammember, p=probabilities, replace=False)

            team_candidate = []
            for i in range(len(selected_indices)):
                team_candidate.append(f"Scientist{selected_indices[i]}")

            print(team_candidate)
            team_list[agent_index][0].log_dialogue(scientists[agent_index].name,','.join(team_candidate))
            is_contained = False
            for agent_list in team_list:
                for sublist in agent_list:
                    if set(sublist.teammate) == set(team_candidate) and sublist.state != self.over_state:
                        is_contained = True
                        break
                if is_contained == True:
                    break
            if is_contained == True:
                continue
            # ask each scientist to decide whether to join
            agent_candidate = self.id_to_agent(team_candidate)
            # create new team
            team_index = []
            team_index.append(scientists[agent_index].name)
            for agent in agent_candidate:
                if agent.name == scientists[agent_index].name:
                    continue
                hint = self.HostMsg(content=Prompts.to_scientist_choice.format_map({
                    "inviter_name": scientists[agent_index].name,
                    "team_member": str(team_index),
                    "personal information" : convert_you_to_other(scientists[agent_index].sys_prompt)
                }))
                # set_parsers(agent, Prompts.scientist_invite_parser)
                pattern = re.compile(r'action\s*1', re.IGNORECASE)
                # action1 means a scientist accepts the invitance
                x = agent.reply(hint, use_memory=False, use_RAG=False)
                if pattern.search(extract_between_json_tags(x.content,num=1)):
                    team_index.append(agent.name)
                team_list[agent_index][0].log_dialogue('user',hint.content)
                team_list[agent_index][0].log_dialogue(agent.name,x.content)
            # delete repeated teams
            is_contained = False
            for agent_list in team_list:
                for sublist in agent_list:
                    if set(sublist.teammate) == set(team_index) and sublist.state != self.over_state:
                        is_contained = True
                        break
                if is_contained == True:
                    break
            if is_contained == False:
                team_dic = Team(team_name = str(agent_index+1)+','+str(len(self.team_pool[agent_index])+1),
                                log_dir = self.log_dir,
                                info_dir = self.info_dir)
                team_dic.state=2
                team_dic.teammate = team_index
                team_list[agent_index].append(team_dic)

                # connetion between collaborators will be closer
                for member in team_dic.teammate:
                    if int(member[9:])!=agent_index:
                        self.adjacency_matrix[agent_index,int(member[9:])]=self.adjacency_matrix[agent_index,int(member[9:])]+0.2
                        self.adjacency_matrix[int(member[9:]),agent_index]=self.adjacency_matrix[int(member[9:]),agent_index]+0.2
                # summary current teams in memory
                scientists[agent_index].prompt_reply(self.HostMsg(content=team_description_detail(team_list[agent_index], self.agent_pool, self.over_state)))
            else:
                continue
        return team_list
    # ===diversity===
    # def select_coauthors(self):
    #     team_list = self.team_pool
    #     scientists = self.agent_pool[:self.agent_num]
    #
    #     for agent_index in range(len(scientists)):
    #         if count_team(team_list[agent_index], self.over_state) >= self.team_limit:
    #             if team_list[agent_index][0].state == 1:
    #                 team_list[agent_index][0].state = 2
    #             continue
    #
    #         scientists[agent_index].sys_prompt += Prompts.role
    #         hint = self.HostMsg(content=Prompts.ask_choice.format_map({
    #             "Scientist_name": scientists[agent_index].name,
    #             "All_team": team_description(team_list[agent_index], self.over_state)
    #         }))
    #
    #         x = scientists[agent_index].reply(hint)
    #         team_list[agent_index][0].log_dialogue('user', hint.content)
    #         team_list[agent_index][0].log_dialogue(scientists[agent_index].name, x.content)
    #         match = re.search(r'action\s*(\d+)', extract_between_json_tags(x.content, num=1), re.IGNORECASE)
    #
    #         if match is not None and int(match.group(1)) == 2:
    #             print("Single Agent Independently!")
    #             team_list[agent_index][0].state = 2
    #             continue
    #
    #         scientist = scientists[agent_index].name
    #         name = int(scientist[9:])
    #         arr = self.adjacency_matrix[name, :]
    #         arr[agent_index] = 0
    #         probabilities = arr / np.sum(arr)
    #
    #         # 强制选择特定 agent（例如 scientist1）
    #         forced_agent = scientists[agent_index].name  # 假设要强制选择的 agent 是当前 agent
    #         selected_indices = np.random.choice(len(arr), size=self.max_teammember - 2, p=probabilities, replace=True)
    #
    #         # 构建团队候选人，强制加入同一个 agent 两次
    #         team_candidate = [forced_agent, forced_agent, forced_agent] + [f"Scientist{selected_indices[i]}" for i in
    #                                                          range(len(selected_indices))]
    #
    #         # 过滤 team_candidate，确保所有 agent 都在 self.id2agent 中
    #         team_candidate = [agent for agent in team_candidate if agent in self.id2agent]
    #
    #         print(team_candidate)
    #         team_list[agent_index][0].log_dialogue(scientists[agent_index].name, ','.join(team_candidate))
    #
    #         # 检查团队是否已经存在
    #         is_contained = False
    #         for agent_list in team_list:
    #             for sublist in agent_list:
    #                 if set(sublist.teammate) == set(team_candidate) and sublist.state != self.over_state:
    #                     is_contained = True
    #                     break
    #             if is_contained:
    #                 break
    #         if is_contained:
    #             continue
    #
    #         # 邀请成员加入团队
    #         agent_candidate = self.id_to_agent(team_candidate)
    #         team_index = [scientists[agent_index].name]
    #
    #         for agent in agent_candidate:
    #             if agent.name == scientists[agent_index].name:
    #                 continue
    #             hint = self.HostMsg(content=Prompts.to_scientist_choice.format_map({
    #                 "inviter_name": scientists[agent_index].name,
    #                 "team_member": str(team_index),
    #                 "personal information": convert_you_to_other(scientists[agent_index].sys_prompt)
    #             }))
    #             pattern = re.compile(r'action\s*1', re.IGNORECASE)
    #             x = agent.reply(hint, use_memory=False, use_RAG=False)
    #             if pattern.search(extract_between_json_tags(x.content, num=1)):
    #                 team_index.append(agent.name)
    #             team_list[agent_index][0].log_dialogue('user', hint.content)
    #             team_list[agent_index][0].log_dialogue(agent.name, x.content)
    #
    #         # 检查是否已有相同团队
    #         is_contained = False
    #         for agent_list in team_list:
    #             for sublist in agent_list:
    #                 if set(sublist.teammate) == set(team_index) and sublist.state != self.over_state:
    #                     is_contained = True
    #                     break
    #             if is_contained:
    #                 break
    #         if not is_contained:
    #             team_dic = Team(team_name=str(agent_index + 1) + ',' + str(len(self.team_pool[agent_index]) + 1),
    #                             log_dir=self.log_dir,
    #                             info_dir=self.info_dir)
    #             team_dic.state = 2
    #             team_dic.teammate = team_index
    #             team_list[agent_index].append(team_dic)
    #
    #             # 更新邻接矩阵
    #             for member in team_dic.teammate:
    #                 if int(member[9:]) != agent_index:
    #                     self.adjacency_matrix[agent_index, int(member[9:])] += 0.2
    #                     self.adjacency_matrix[int(member[9:]), agent_index] += 0.2
    #
    #             # 更新当前团队信息
    #             scientists[agent_index].prompt_reply(self.HostMsg(
    #                 content=team_description_detail(team_list[agent_index], self.agent_pool, self.over_state)))
    #         else:
    #             continue
    #     return team_list

    def id_to_agent(self, team_list):
        agent_list = []
        for agent_id in team_list:
            agent_list.append(self.id2agent[agent_id])
        return agent_list

    def agent_to_id(self, team_list):
        agent_list = []
        for agent_id in team_list:
            agent_list.append(agent_id.name)
        return agent_list

    def reference_paper(self, key_string, cite_number):
        query_vector = ollama.embeddings(model="mxbai-embed-large", prompt=key_string) # 使用ollama库为输入关键词生成嵌入向量 query_vector
        query_vector = np.array([query_vector['embedding']], dtype=np.float32)
        D, I = self.gpu_index.search(query_vector, cite_number) # 使用self.gpu_index.search（向量数据库搜索函数）查找与查询向量最近的cite_number个邻居。D包含距离，I包含self.paper_dicts中最近邻的索引

        # 新加的警告
        # valid_indices = []  # 在此处初始化 valid_indices
        # if len(I[0]) > 0:  # 在处理之前检查是否有结果
        #     valid_indices = [i for i in I[0] if 0 <= i < len(self.paper_dicts)]
        #     if len(valid_indices) < cite_number:
        #         print(f"警告：Faiss 搜索返回的有效索引少于 {cite_number} 个。使用 {len(valid_indices)} 个。")

        paper_use = []
        for id in range(len(I[0])):
            paper_title = self.paper_dicts[I[0][id]]['title']
            paper_abstract = self.paper_dicts[I[0][id]]['abstract']
            paper_index = {}
            paper_index['title'] = paper_title
            paper_index['abstract'] = paper_abstract # 代码迭代I中的索引，从self.paper_dicts中检索论文标题和摘要。
            paper_use.append(paper_index)
        # for index in valid_indices:  # 使用 valid_indices
        #     try:
        #         paper_title = self.paper_dicts[index]['title']
        #         paper_abstract = self.paper_dicts[index]['abstract']
        #         paper_use.append({'title': paper_title, 'abstract': paper_abstract})
        #     except (KeyError, IndexError) as e:
        #         print(f"处理索引为 {index} 的论文时出错：{e}")
        #         continue # 如果出错，跳过这篇论文
        paper_reference = ""  # 将结果格式化为字符串
        for id in range(len(paper_use)):
            paper_index = paper_use[id]
            paper_reference = paper_reference+"Paper {}:".format(id+1)+"\n"
            paper_reference = paper_reference+"Title: "+paper_index['title']+"\n"
            paper_reference = paper_reference+"Abstract: "+paper_index['abstract']+"}"+"\n"
        return paper_reference, I[0]

    # ===正确的===
    def running(self, epochs):
        # init team_pool
        print(f'Epoch{-1}-------------------initialize team')
        self.team_pool = self.select_coauthors()
        for epoch in range(epochs):
            # state 7 is an over
            # 1. select coauthors for state 1
            # 2. select topics for state 2
            # 3. generate idea for state 3
            # 4. check novelty for state 4
            # 5. generate paper abstract for state 5
            # 6. generate paper review for state 6
            # 对每个团队执行操作
            for leader_index in range(len(self.team_pool)):
                for team_index in range(len(self.team_pool[leader_index])):
                    self.team_pool[leader_index][team_index].epoch = epoch
                    self.team_pool[leader_index][team_index].action_excution(self)
                    if self.team_pool[leader_index][team_index].state == 7:
                        self.team_pool[leader_index][team_index].save_team_info()

            print(f'Epoch{epoch}-------------------begin select authors')
            self.team_pool = self.select_coauthors()
            print(f'Epoch{epoch}-------------------current action finished')

    # ===测试背景完全一样agent表现===
    # def running(self, epochs):
    #     # 跳过 select_coauthors 步骤，直接使用初始化时设置的团队
    #     print(f'Epoch{-1}-------------------initialize team')
    #     for epoch in range(epochs):
    #         # 执行团队活动
    #         for leader_index in range(len(self.team_pool)):
    #             for team_index in range(len(self.team_pool[leader_index])):
    #                 self.team_pool[leader_index][team_index].epoch = epoch
    #                 self.team_pool[leader_index][team_index].action_excution(self)
    #                 if self.team_pool[leader_index][team_index].state == 7:
    #                     self.team_pool[leader_index][team_index].save_team_info()
    #
    #         print(f'Epoch{epoch}-------------------current action finished')