var tree = {
    draw:function(id,data){
        var option = this.getOption(data)
        var myChart = echarts.init(document.getElementById(id));
        myChart.setOption(option)
    },
    getResult:function(result){
        switch(result){
            case 0:
                return "中性"
            case 1:
                return "正面"
            case -1:
                return "负面"
            case -2:
                return "未提及"
        }
    },
    proccessData:function(data){
        var that = this
        return data_option = {
            "name": "被分析文本",
            "children": [
                {
                    "name": "交通", 
                    "children":[{
                        "name":"交通是否便利",
                        "children":[{
                                "name":that.getResult(data.location_traffic_convenience)//指标结果
                            }]
                    },{
                        "name":"距离商圈远近",
                        "children":[{
                                "name":that.getResult(data.location_distance_from_business_district)//指标结果
                            }]
                    },{
                        "name":"是否容易找到"  ,
                        "children":[{
                                "name":that.getResult(data.location_easy_to_find)//指标结果
                            }]
                    }]
                    
                },
                {
                    "name": "服务",
                    "children":[{
                        "name":"排队等候时间",
                        "children":[{
                                "name":that.getResult(data.service_wait_time)//指标结果
                            }]   
                    },{
                        "name":"服务人员态度",
                        "children":[{
                                "name":that.getResult(data.service_waiters_attitude)//指标结果
                            }]   
                    },{
                        "name":"是否容易停车",
                        "children":[{
                                "name":that.getResult(data.service_parking_convenience)//指标结果
                            }]  
                    },{
                        "name":"点菜/上菜速度" ,
                        "children":[{
                                "name":that.getResult(data.service_serving_speed)//指标结果
                            }] 
                    }]
                    
                },
                {
                    "name": "价格",
                    "children":[{
                        "name":"价格水平" ,
                        "children":[{
                                "name":that.getResult(data.price_level)//指标结果
                            }]  
                    },{
                        "name":"性价比" ,
                        "children":[{
                                "name":that.getResult(data.price_cost_effective)//指标结果
                            }]  
                    },{
                        "name":"折扣力度" ,
                        "children":[{
                                "name":that.getResult(data.price_discount)//指标结果
                            }] 
                    }]
                
                },
                {
                    "name": "环境",
                    "children":[{
                        "name":"装修情况"  ,
                        "children":[{
                                "name":that.getResult(data.environment_decoration)//指标结果
                            }] 
                    },{
                        "name":"嘈杂情况",
                        "children":[{
                                "name":that.getResult(data.environment_noise)//指标结果
                            }]   
                    },{
                        "name":"就餐空间",
                        "children":[{
                                "name":that.getResult(data.environment_space)//指标结果
                            }]  
                    },{
                        "name":"卫生情况" ,
                        "children":[{
                                "name":that.getResult(data.environment_cleaness)//指标结果
                            }]   
                    }]
                    
                },
                {
                    "name": "菜品",
                    "children":[{
                        "name":"分量",
                        "children":[{
                                "name":that.getResult(data.dish_portion)//指标结果
                            }]     
                    },{
                        "name":"口感"  ,
                        "children":[{
                                "name":that.getResult(data.dish_taste)//指标结果
                            }]   
                    },{
                        "name":"外观",
                        "children":[{
                                "name":that.getResult(data.dish_look)//指标结果
                            }]    
                    },{
                        "name":"推荐程度",
                        "children":[{
                                "name":that.getResult(data.dish_recommendation)//指标结果
                            }]    
                    }]
                    
                },
                {
                    "name": "其他",
                    "children":[{
                        "name":"本次消费感受",
                        "children":[{
                                "name":that.getResult(data.others_overall_experience)//指标结果
                            }]     
                    },{
                        "name":"再次消费意愿" ,
                        "children":[{
                                "name":that.getResult(data.others_willing_to_consume_again)//指标结果
                            }]    
                    }]
                    
                }
            ]
        };
        
    },
    getOption:function(data){
        var data_option = this.proccessData(data)
        return option = {
            // title: {
            //     text: '树图分析'
            // },
            tooltip: {
                trigger: 'item',
                triggerOn: 'mousemove'
            },
            series:[
                {
                    type: 'tree',
                    id: 0,
                    name: 'tree1',
                    data: [data_option],

                    top: '10%',
                    left: 70,
                    bottom: '10%',

                    symbolSize: 7,

                    edgeShape: 'polyline',
                    edgeForkPosition: '63%',
                    initialTreeDepth: 3,

                    lineStyle: {
                        width: 2
                    },

                    label: {
                        backgroundColor: '#fff',
                        position: 'left',
                        verticalAlign: 'middle',
                        align: 'right'
                    },

                    leaves: {
                        label: {
                            position: 'right',
                            verticalAlign: 'middle',
                            align: 'left'
                        }
                    },

                    expandAndCollapse: true,
                    animationDuration: 550,
                    animationDurationUpdate: 750
                }
            ]
        };
    },
}