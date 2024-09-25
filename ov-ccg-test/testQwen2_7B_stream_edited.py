import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time
import argparse

from transformers import AutoTokenizer, TextStreamer
import numpy as np

sentences = [
    "What is OpenVINO?",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",    
    "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
    "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content.",
    "患者男，年龄29岁，血型O，因思维迟钝，易激怒，因发热伴牙龈出血14天，乏力、头晕5天就诊我院急诊科。快速完善检查，血常规显示患者三系血细胞重度减低，凝血功能检查提示APTT明显延长，纤维蛋白原降低，血液科会诊后发现患者高热、牙龈持续出血，胸骨压痛阳性.于3903年3月7日入院治疗，出现头痛、头晕、伴发热（最高体温42℃）症状，曾到其他医院就医。8日症状有所好转，9日仍有头痛、呕吐，四肢乏力伴发热。10日凌晨到本院就诊。患者5d前出现突发性思维迟钝，脾气暴躁，略有不顺心就出现攻击行为，在院外未行任何诊治。既往身体健康，平素性格内向。体格检查无异常。血常规白细胞中单核细胞百分比升高。D-二聚体定量1412μg/L，骨髓穿刺示增生极度活跃，异常早幼粒细胞占94%.外周血涂片见大量早幼粒细胞，并可在胞浆见到柴捆样细胞.以下是血常规详细信息：1.病人红细胞计数结果：3.2 x10^12/L. 附正常参考范围：新生儿:（6.0～7.0）×10^12/L；婴儿：（5.2～7.0）×10^12/L; 儿童：（4.2～5.2）×10^12/L; 成人男：（4.0～5.5）×10^12/L; 成人女：（3.5～5.0）×10^12/L. 临床意义：生理性红细胞和血红蛋白增多的原因：精神因素（冲动、兴奋、恐惧、冷水浴刺激等导致肾上腺素分泌增多的因素）、红细胞代偿性增生（长期低气压、缺氧刺激，多次献血）；生理性红细胞和血红蛋白减少的原因：造血原料相对不足，多见于妊娠、6个月～2岁婴幼儿、某些老年性造血功能减退；病理性增多：多见于频繁呕吐、出汗过多、大面积烧伤、血液浓缩，慢性肺心病、肺气肿、高原病、肿瘤以及真性红细胞增多症等；病理性减少：多见于白血病等血液系统疾病；急性大出血、严重的组织损伤及血细胞的破坏等；合成障碍，见于缺铁、维生素B12缺乏等。2. 病人血红蛋白测量结果：108g/L. 附血红蛋白正常参考范围：男性120～160g/L；女性110～150g/L；新生儿170～200g/L；临床意义：临床意义与红细胞计数相仿，但能更好地反映贫血程度，极重度贫血（Hb<30g/L）、重度贫血（31～60g/L）、中度贫血（61～90g/L）、男性轻度贫血（90~120g/L）、女性轻度贫血（90~110g/L）。3. 病人白细胞计数结果：13.6 x 10^9/L; 附白细胞计数正常参考范围：成人（4.0～10.0）×10^9/L；新生儿（11.0～12.0）×10^9/L。临床意义：1）生理性白细胞计数增高见于剧烈运动、妊娠、新生儿；2）病理性白细胞增高见于急性化脓性感染、尿毒症、白血病、组织损伤、急性出血等；3）病理性白细胞减少见于再生障碍性贫血、某些传染病、肝硬化、脾功能亢进、放疗化疗等。4. 病人白细胞分类技术结果：中性粒细胞（N）50%、嗜酸性粒细胞（E）3.8%、嗜碱性粒细胞（B）0.2%、淋巴细胞（L）45%、单核细胞（M）1%。附白细胞分类计数正常参考范围：中性粒细胞（N）50%～70%、嗜酸性粒细胞（E）1%～5%、嗜碱性粒细胞（B）0～1%、淋巴细胞（L）20%～40%、单核细胞（M）3%～8%；临床意义：1）中性粒细胞为血液中的主要吞噬细胞，在细菌性感染中起重要作用。2）嗜酸性粒细胞①减少见于伤寒、副伤寒、大手术后、严重烧伤、长期用肾上腺皮质激素等。②增多见于过敏性疾病、皮肤病、寄生虫病，一些血液病及肿瘤，如慢性粒细胞性白血病、鼻咽癌、肺癌以及宫颈癌等；3）嗜碱性粒细胞 a 减少见于速发型过敏反应如过敏性休克，肾上腺皮质激素使用过量等。b 增多见于血液病如慢性粒细胞白血病，创伤及中毒，恶性肿瘤，过敏性疾病等；4）淋巴细胞 a 减少多见于传染病的急性期、放射病、细胞免疫缺陷病、长期应用肾上腺皮质激素后或放射线接触等。b 增多见于传染性淋巴细胞增多症、结核病、疟疾、慢性淋巴细胞白血病、百日咳、某些病毒感染等；5）单核细胞增多见于传染病或寄生虫病、结核病活动期、单核细胞白血病、疟疾等。5. 病人血小板计数结果：91 x10^9/L. 附血小板计数正常参考范围：（100～300）×10^9/L. 临床意义：1）血小板计数增高见于真性红细胞增多症、出血性血小板增多症、多发性骨髓瘤、慢性粒细胞性白血病及某些恶性肿瘤的早期等；2）血小板计数减低见于 a 骨髓造血功能受损，如再生障碍性贫血，急性白血病；b 血小板破坏过多，如脾功能亢进；c 血小板消耗过多，如弥散性血管内凝血等。6. 以往病例分析内容参考：白血病一般分为急性白血病和慢性白血病。1）急性白血病血常规报告表现为：白细胞增高，少数大于100×10^9/L，称为高白细胞白血病，部分患者白细胞正常或减少，低者可小于1.0×10^9/L，以AML中的M3型多见。在白细胞分类中，80％以上可见大量的幼稚细胞，有时仅见幼稚细胞和少量成熟的细胞，而无中间型细胞，称为白血病的裂孔现象。少数白细胞低的患者周围血幼稚细胞很少，此类患者必须骨髓穿刺才能确诊。多数急性白血病患者初诊时有不同程度的贫血；一般属正常细胞正色素型。但贫血很快会进行性加重。30％的患者血涂片中可见有核红细胞。血小板计数绝大部分患者减少，严重者小于10×10^9/L，仅极少数患者血小板计数正常。2） 慢性白血病血常规报告表现为：白细胞总数明显增高，通常大于30×10^9/L。半数患者大于100×10^9/L。中性粒细胞明显增多，可见各阶段粒细胞，以中性中幼粒，晚幼粒细胞居多，原始粒细胞小于等于10％，通常为1％～5％，嗜酸和嗜碱粒细胞亦增多。初诊时约有50％患者血小板增高，少数可大于1000×10^9/L。红细胞和血红蛋白一般在正常范围，若出现血红蛋白减低，血小板计数明显升高或降低，则提示疾病向加速期或急变期转化。7. 历史相关研究: 急性髓系白血病（AML）是造血干细胞恶性克隆性疾病。在AML的诊断、治疗以及判断预后的过程中，基因异常是一项重要指标。随着基因检测技术的不断进步，越来越多与AML发生相关的基因被人们发现，并且这些基因在指导预后方面有重要意义。常见和急性髓系白血病相关的基因突变有: 1）RUNX1-RUNX1T1; 2）CBFB-MYH11; 3）NPM1：核磷蛋白1（nucleophosmin 1，NPM1）; 4）CEBPA：CCAAT增强子结合蛋白α基因（CCAAT/en－hancer binding protein α，CEBPA）; 5）MLLT3-KMT2A; 6）DEK-NUP214; 7）KMT2A：KMT2A基因（也称为MLL基因）问：请基于以上信息做出判断，该患者是否有罹患急性白血病的风险？请结合上述内容给出判断的详细解释，并简要总结潜在的早期征兆、预防方法、相关的基因突变、常用的治疗手段，以及当前已上市和正在临床阶段的药物清单。答：",
    # "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    # "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
]

sentence_long = open("longbench_4k.txt", 'r', encoding='utf-8').read()
sentences.append(sentence_long)
sentences.append(sentence_long)


class LatencyTrackingStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False):
        super().__init__(tokenizer, skip_prompt)
        self.skip_prompt_jump = skip_prompt
        self.start_time = None
        self.first_token_time = None

    def clean(self):
        self.skip_prompt_jump = True
        self.start_time = None
        self.first_token_time = None

    def put(self, value):
        if self.first_token_time is None:
            if self.skip_prompt_jump == True:
                self.skip_prompt_jump = False
            else:
                # torch.xpu.synchronize()
                self.first_token_time = time.time()  # 记录第一个token的时间
        super().put(value)

    def start_tracking(self):
        self.start_time = time.time()  # 记录生成命令发下去的时间
        
    def recorded_time(self):
        if self.start_time is not None and self.first_token_time is not None: 
            return self.first_token_time - self.start_time
        else:
            return None
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Qwen2-7B-Instruct')
    parser.add_argument('--repo-id-or-model-path', type=str, default=r"C:\Users\v_52\Downloads\ipex-llm-test\Qwen2-7B-Instruct",
                        help='The huggingface repo id for the Qwen2 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="AI是什么？",
                        help='Prompt to infer') 
    parser.add_argument('--n-predict', type=int, default=512,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    ov_config = {"PERFORMANCE_HINT": "LATENCY",
                 "NUM_STREAMS": "1", "CACHE_DIR": ""}
    # model_path = "./Qwen2-7B-Instruct-ov-sym-int4-1.0"

    from optimum.intel import OVModelForCausalLM
    from transformers import AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(
        model_path)
    print("====Compiling model====")
    model = OVModelForCausalLM.from_pretrained(
        model_path,
        device="GPU",
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_path),
    )

    prompt = args.prompt

    # Generate predicted tokens
    with torch.inference_mode():
        # The following code for generation is adapted from https://huggingface.co/Qwen/Qwen2-1.5B-Instruct#quickstart
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
        model_inputs = tokenizer([text], return_tensors="pt")#.to("xpu")
        # warmup
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.n_predict
            )
        
        for idx, prompt in enumerate(sentences):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )
            model_inputs = tokenizer([text], return_tensors="pt")#.to("xpu")
            input_ids = model_inputs.input_ids
            # deal with 3k/4k input
            if idx == len(sentences) - 2:
                in_len = 3072
                half_idx = in_len // 2
                input_ids = \
                    torch.cat((model_inputs.input_ids[:, :half_idx], model_inputs.input_ids[:, -(in_len-half_idx):]), dim=1)
            elif idx == len(sentences) - 1:
                in_len = 4096
                half_idx = in_len // 2
                input_ids = \
                    torch.cat((model_inputs.input_ids[:, :half_idx], model_inputs.input_ids[:, -(in_len-half_idx):]), dim=1)


            print("---Start warmup---")
            # warmup for each trail
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=args.n_predict,
                do_sample=False,
            )
            print("---Warmup finish---")

            # streamer = TextStreamer(tokenizer, skip_prompt=True)
            streamer = LatencyTrackingStreamer(tokenizer, skip_prompt=True)
            first_token_list = []
            rest_token_list = []
            for i in range(3):

                # 在生成之前开始跟踪时间
                streamer.clean()
                streamer.start_tracking()

                st = time.time()
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.n_predict, 
                    streamer=streamer, 
                    do_sample=False,
                    )
                # torch.xpu.synchronize()
                end = time.time()
                generated_ids = generated_ids.cpu()
                generated_ids = [
                    output_token_ids[len(input_token_ids):] for input_token_ids, output_token_ids in zip(input_ids, generated_ids)
                    ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                first_token_latency = 1000*streamer.recorded_time()
                rest_token_latency = 1000*(end-st-streamer.recorded_time())/(len(generated_ids[0])-1)

                print('-'*30, f'Trial {i}', '-'*30)
                print(f'    Input tokens: {len(input_ids[0])}')
                print(f'    Output tokens: {len(generated_ids[0])}')
                print(f"    1st token latency: {first_token_latency:.6f} ms")
                print(f'    Average latency: {rest_token_latency:.6f} ms/t')
                print('-'*30)

                first_token_list.append(first_token_latency)
                rest_token_list.append(rest_token_latency)

                print('-'*20, 'perf mode info', '-'*20)
                if getattr(model, 'n_drafted', None) is not None:
                    draft_len = model.n_drafted/len(model.draft_num)
                    accept_rate = model.n_matched/model.n_drafted
                    print(f"Draft Number: {model.draft_num}")
                    print(f"Accept Number: {model.accept_num}")
                    print(f"Draft len: {draft_len}")
                    print(f"Accept len: {model.n_matched/len(model.accept_num)}")
                    print(f"Accept rate: {accept_rate*100}%")
                    print(f"1st token: {model.first_token_time} ms")
                    print(f"2+ token: {(end - st - model.first_token_time)/(model.n_token_generated - 1)} ms")
                print('-'*50)

            # print(f'Inference time: {end-st} s')
            print('='*60)
            print(f'    Input tokens: {len(input_ids[0])}')
            print(f'    Output tokens: {len(generated_ids[0])}')
            print(f"    1st token latency list: {first_token_list} ms")
            print(f"    1st token latency avg.: {(sum(first_token_list)/len(first_token_list)):.6f} ms")
            print(f"    Rest token latency list: {rest_token_list} ms")
            print(f"    Rest token latency avg.: {(sum(rest_token_list)/len(rest_token_list)):.6f} ms")
            print('='*60)
            print('-'*20, 'Prompt', '-'*20)
            print(prompt)
            print('-'*20, 'Output', '-'*20)
            print(response)