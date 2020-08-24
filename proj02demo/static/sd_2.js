var sd_2 = {
    draw:function(id,data){
        var option = this.getOption(data)
        var myChart = echarts.init(document.getElementById(id));
        myChart.setOption(option)
    },
    getResultIndex:function(result){//index--0，1，2，3分别代表未提及 负面 中性 正面；
        switch(result){
            case 0:
                return 2
            case 1:
                return 3
            case -1:
                return 1
            case -2:
                return 0
        }
    },
    proccessData:function(data){
        var that = this;
        var data_value = []
        var attrs = []
        for(var attr in data){
            attrs.push(attr)
        }
        for (var i = 0; i <20; i++){
            var index = that.getResultIndex(data[attrs[i]])
            for (var j = 0; j<4; j++){
                var temp = [i,j]
                j ==index?temp.push(4):temp.push(0)
                data_value.push(temp)
            }

        }
        return data_value
        
    },
    getOption:function(data){
        var that = this
        var data_value = this.proccessData(data)
        var category = ['交通是否便利', '距离商圈远近', '是否容易找到',
        '排队等候时间', "服务人员态度",'是否容易停车','点菜/上菜速度', 
        '价格水平', '性价比','折扣力度',
        '装修情况', '嘈杂情况', '就餐空间','卫生情况', 
        '分量', '口感','外观', '推荐程度', 
        '本次消费感受','再次消费意愿'];//20中评价指标，用于显示

        var zhibiao = ['未提及','负面','中性','正面'];

        var option = {
            tooltip: {
                position: 'top'
            },
            title: [],
            singleAxis: [],
            series: []
        };

        echarts.util.each(category, function (c, idx) {
            option.title.push({
                textBaseline: 'middle',
                top: (idx + 0.5) * 100 / 20 + '%',
                text: c
            });
            option.singleAxis.push({
                left: 300,
                type: 'category',
                boundaryGap: false,
                data: zhibiao,
                top: (idx * 100 / 20 )+2 + '%',
                height: (100 / 20 - 7) + '%',
                axisLabel: {
                    interval: 0,
                    show:true,
                    
                    showMinLabel:true,
                }
            });
            option.series.push({
                singleAxisIndex: idx,
                coordinateSystem: 'singleAxis',
                type: 'scatter',
                data: [],
                symbolSize: function (dataItem) {
                    return dataItem[1] * 4;
                }
            });
        });

        echarts.util.each(data_value, function (dataItem) {
            option.series[dataItem[0]].data.push([dataItem[1], dataItem[2]]);
        });

        return option
    },
}