# coding=utf-8
import numpy as np
import sys, json
import pandas as pd
import random

def extract_text(data):
    data2text = {
        "location_traffic_convenience": {-1:"交通不便", 0:"交通一般", 1:"交通方便"},
        "location_distance_from_business_district": {-1:"距离商业区远", 0:"距离商区正常", 1:"距离商区近"},
        "location_easy_to_find": {-1:"位置难找", 0:"位置一般", 1:"位置好找"},
        "service_wait_time": {-1:"排队等待时间长", 0:"排队等待时间还行", 1:"排队等待时间短"},
        "service_waiters_attitude": {-1:"服务态度差", 0:"服务态度还行", 1:"服务态度好"},
        "service_parking_convenience": {-1:"停车不便", 0:"停车还行", 1:"停车方便"},
        "service_serving_speed": {-1:"上菜速度慢", 0:"上菜速度还行", 1:"上菜速度快"},
        "price_level": {-1:"价格贵", 0:"价格还行", 1:"价格便宜"},
        "price_cost_effective": {-1:"性价比低", 0:"性价比还行", 1:"性价比高"},
        "price_discount": {-1:"折扣太少", 0:"折扣还行", 1:"折扣力度大"},
        "environment_decoration": {-1:"装修差", 0:"装修一般", 1:"装修好"},
        "environment_noise": {-1:"环境嘈杂", 0:"环境嘈杂程度还行", 1:"环境安静"},
        "environment_space": {-1:"空间小", 0:"空间还行", 1:"空间大"},
        "environment_cleaness": {-1:"整洁程度较差", 0:"整洁程度还行", 1:"整洁程度好"},
        "dish_portion": {-1:"分量少", 0:"分量还行", 1:"分量足"},
        "dish_taste": {-1:"味道差", 0:"味道正常", 1:"味道好"},
        "dish_look": {-1:"外观不好看", 0:"外观还行", 1:"外观好看"},
        "dish_recommendation": {-1:"不推荐来", 0:"推荐程度中等", 1:"推荐来"},
        "others_overall_experience": {-1:"总体不行", 0:"总体还行", 1:"总体很好"},
        "others_willing_to_consume_again": {-1:"下次不会来了", 0:"下次不一定会再来", 1:"下次还会来"}
    }
    meta = {"meta":{"positive":[], "neutral":[], "negative":[]}}
    for k, v in data.items():
        if v != -2: 
            if v == -1: meta["meta"]["negative"].append(data2text.get(k).get(v))
            if v == 0: meta["meta"]["neutral"].append(data2text.get(k).get(v))
            if v == 1: meta["meta"]["positive"].append(data2text.get(k).get(v))
    return meta
        
def give_demo():
    demo_data = {
        "result":{
            "location_traffic_convenience": np.random.randint(4)-2,
            "location_distance_from_business_district": np.random.randint(4)-2,
            "location_easy_to_find": np.random.randint(4)-2,
            "service_wait_time": np.random.randint(4)-2,
            "service_waiters_attitude": np.random.randint(4)-2,
            "service_parking_convenience": np.random.randint(4)-2,
            "service_serving_speed": np.random.randint(4)-2,
            "price_level": np.random.randint(4)-2,
            "price_cost_effective": np.random.randint(4)-2,
            "price_discount": np.random.randint(4)-2,
            "environment_decoration": np.random.randint(4)-2,
            "environment_noise": np.random.randint(4)-2,
            "environment_space": np.random.randint(4)-2,
            "environment_cleaness": np.random.randint(4)-2,
            "dish_portion": np.random.randint(4)-2,
            "dish_taste": np.random.randint(4)-2,
            "dish_look": np.random.randint(4)-2,
            "dish_recommendation": np.random.randint(4)-2,
            "others_overall_experience": np.random.randint(4)-2,
            "others_willing_to_consume_again": np.random.randint(4)-2
        }
        
    }

    demo_data.update(extract_text(demo_data["result"]))
    # print(demo_data)
    return json.dumps(demo_data)

def give_demo2():
    a = json.loads(give_demo())
    a["text"] = "这是一段随机文本"+str(np.random.randint(10000))
    return json.dumps(a)



# trainingset_path=None
# trainingset = pd.read_csv(trainingset_path,encoding='utf-8')
# labels = list(trainingset.columns.values)
# labels.pop(0)
class FeelingLucky:
    def __init__(self, path='./dataset/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'):
        self.df = pd.read_csv(path, encoding="utf-8", index_col=0)
        
    def feelinglucky(self, ):
        # choice = random.choice(self.df)
        output = {}
        raw = json.loads(self.df.sample().to_json(orient="records"))[0]
        # try:
        #     output["text"] = eval(raw.pop("content"))
        # except:
        output["text"] = raw.pop("content")[1:-1]
        output["result"] = raw
        output.update(extract_text(output["result"]))
        print(output)
        return output



if __name__ == "__main__":
    # give_demo()
    fl = FeelingLucky()
    fl.feelinglucky()