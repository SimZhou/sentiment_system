var leida = {
    draw:function(id,data){
        var option = this.getOption(data)
        var myChart = echarts.init(document.getElementById(id));
        myChart.setOption(option)
    },

    proccessData:function(data){
        var data_leida ={
            "location":{
                "traffic conveenience":data.location_traffic_convenience,
                "distance from business distric":data.location_distance_from_business_district,
                "easy to find" :data.location_easy_to_find
            },
            "service":{
                "wait time":data.service_wait_time,
                "waiter's attitude":data.service_waiters_attitude,
                "parking convenience":data.service_parking_convenience,
                "serving speed":data.service_serving_speed,
            },
            "price":{
                "price level" :data.price_level,
                "cost effective":data.price_cost_effective,
                "discount":data.price_discount,
            },
            "environment":{
                "decoration":data.environment_decoration,
                "noise":data.environment_noise,
                "space":data.environment_space,
                "cleaness":data.environment_cleaness,
            },
            "dish":{
                "portion":data.dish_portion,
                "tast":data.dish_taste,
                "look":data.dish_look,
                "recommendation":data.dish_recommendation,
            },
            "others":{
                "overall experience":data.others_overall_experience,
                "willing to consume aggain":data.others_willing_to_consume_again,
            }
        }
        var data_value = []
        for (var attr in data_leida){
            var value = 0
            for (var attrr in data_leida[attr]){
                value = value + data_leida[attr][attrr]+3
            }
            data_value.push(value*1000)
        }
        return data_value
    },
    getOption:function(data){
        var data_value = this.proccessData(data)
        return option = {
            // title: {
            //     text: '雷达图分析'
            // },
            tooltip: {
            },
            radar: {
                // shape: 'circle',
                name: {
                    textStyle: {
                        color: '#fff',
                        backgroundColor: '#999',
                        borderRadius: 3,
                        padding: [3, 5]
                    }
                },
                indicator: [
                    { name: '位置', max: 20000},
                    { name: '服务', max: 20000},
                    { name: '价格', max: 20000},
                    { name: '环境', max: 20000},
                    { name: '菜品', max: 20000},
                    { name: '其他', max: 20000}
                ]
            },
            series: [{
                name: '被分析文本',
                type: 'radar',
                left:0,
                top:"25%",
                bottom:"25%",
                right:0,
                // areaStyle: {normal: {}},
                data: [
                    {
                        value: data_value,
                        name: '被分析文本'
                    },
                ]
            }]
        };
    },

}