var sd_1 = {
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
        var data_value =  [
            [0,0,data.location_traffic_convenience],[0,1,data.location_distance_from_business_district],[0,2,data.location_easy_to_find],//单元组最后一个值是每隔指标的占比
            [0,3,data.service_wait_time],[0,4,data.service_waiters_attitude],[0,5,data.service_parking_convenience],[0,6,data.service_serving_speed],
            [0,7,data.price_level],[0,8,data.price_cost_effective],[0,9,data.price_discount],
            [0,10,data.environment_decoration],[0,11,data.environment_noise],[0,12,data.environment_space],[0,13,data.environment_cleaness],
            [0,14,data.dish_portion],[0,15,data.dish_taste],[0,16,data.dish_look],[0,17,data.dish_recommendation],
            [0,18,data.others_overall_experience],[0,19,data.others_willing_to_consume_again],
        ];
        return data_value

    },
    getOption:function(data){
        var that = this
        var data_value = this.proccessData(data)
        var zhibiao = ['交通是否便利', '距离商圈远近', '是否容易找到',
        '排队等候时间', "服务人员态度",'是否容易停车','点菜/上菜速度', 
        '价格水平', '性价比','折扣力度',
        '装修情况', '嘈杂情况', '就餐空间','卫生情况', 
        '食物分量', '食物味道','食物外观', '推荐程度', 
        '本次消费感受','再次消费意愿'];//20中评价指标，用于显示

        var category = [''];
        var option = {
            tooltip: {
                position: 'top',
                formatter:function(data){
                    var ca = data.data[1]
                    var result = that.getResult(ca)
                    var des = data.name +":"+result
                    return des
                }
                
            },
            title: [],
            singleAxis: [],
            series: [],
            visualMap: {
                    show:true,
                    type: 'piecewise',
                    min: -1,
                    max: 1,
                    splitNumber: 3,
                    inRange: {
                        color: ["#d94e5d","#eac736","#1BAD19"].reverse()
                    },
                    text:["正面","负面"],
                    top: 0,
                    right: -1,
                    // textStyle: {
                    //     color: '#fff'
                    // }
            },
            // color:["#FFB5C5"]
        };

        echarts.util.each(category, function (c, idx) {
            option.title.push({
                textBaseline: 'middle',
                top: (idx + 0.5) * 100 / 10+10 + '%',
                text: c
            });
            option.singleAxis.push({
                left: 70,
                type: 'category',
                boundaryGap: false,
                data: zhibiao,
                top: (idx * 100 / 10 )+10 + '%',
                height: (100 / 10 - 2)+10 + '%',
                axisLabel: {
                    interval: 0,
                    show:true,
                    
                    showMinLabel:true,
                    rotate:30
                }
            });
            option.series.push({
                singleAxisIndex: idx,
                coordinateSystem: 'singleAxis',
                type: 'effectScatter',
                data: [],
                symbolSize: function (dataItem) {
                    if (dataItem[1]==-2){return 0}
                    else {return 12}
                    // return (dataItem[1]+3) * 4;//(+3 是因为有负值，为了保证每个值都大于0)
                },
                itemStyle:{
                    normal:{
                        color: function(dataItem){
                            // switch (dataItem[1]){
                            //     case -1:
                            //         return "#66CD00";
                            //     case 0:
                            //         return "#919191";
                            //     case 1:
                            //         return "#EE0000";
                            // }
                            if (dataItem[1]==-1){return "#32CD32"}
                            else if (dataItem[1]==0){return "#4D4D4D"}
                            else if (dataItem[1]==1){return "#EE2C2C"}
                            else {return "#171717"}
                        }
                    }
                }
            });
        });
        // console.log("middle", data_value, option.series)
        echarts.util.each(data_value, function (dataItem) {
            
            option.series[dataItem[0]].data.push([dataItem[1], dataItem[2]]);
        });
        // option.series.data["visualMap"] = false;
        // return f{{{{{}}}
        console.log("end", option.data, option.series)
        return option
    },
}