import 'dart:async';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';
import 'dart:convert';
import 'package:fl_chart/fl_chart.dart';
import 'dart:math';

import 'package:my_project/connect_django.dart';
import 'package:my_project/screen/screenMain.dart';


class DetailScreen extends StatefulWidget{
  final int pNo;
  DetailScreen({required this.pNo});

  @override
  _DetailScreenState createState() => _DetailScreenState();
}


class _DetailScreenState extends State<DetailScreen> {
  List<FlSpot> actData = []; 
  List<FlSpot> ddotData_blue = [];
  List<FlSpot> ddotData_red = [];

  List<MapEntry<int, String>> labels = [];

  late Timer _timer;

  double minX = 0;
  double maxX = 10800;
  double minY = 0;
  double maxY = 0;
  int dataCount = 0;
  String predStartTime = "";
  String predRecommendedTime = "";

  int? activate_num = null;
  double tempY = 0;
  double tempMinY = 2;
  double tempMaxY = 0;

  List<bool> xLabel_picked =[];

  List<DateTime> date = [];
  List<double> act = [];
  List<bool> pred = [];

  @override
  void initState() {
    super.initState();
    fetchInitialData(); // Custom function to fetch the initial data
    setUpTimedFetch(); // Custom function to set up periodic fetching of data
    
  }

  @override
  void dispose() {
    _timer.cancel(); // Cancel the timer
    super.dispose();  // Always call super.dispose() at the end
  }

  void refreshData() async{
    actData.clear();
    actData.clear();
    ddotData_blue.clear();
    ddotData_red.clear();
    labels.clear();
    date.clear();
    act.clear();
    pred.clear();
    minX = 0;
    maxX = 0;  // Reset these values as per your default or based on new data
    minY = 0;
    maxY = 0;
    dataCount = 0;
    

    // 초기 데이터 가져오기
    fetchInitialData();

    // 상태 업데이트
    setState(() {});
  }


  void addLabelIfNotPresent(int key, String value) {
    // labels 리스트를 순회하며 value 값이 존재하는지 확인합니다.
    bool labelExists = labels.any((entry) => entry.value == value);

    // 해당 String 값이 리스트에 없으면 새로운 MapEntry를 추가합니다.
    if (!labelExists) {
      labels.add(MapEntry(key, value));
    }
  }

  int binarySearch(List<int> sortedList, int value) {
    int low = 0;
    int high = sortedList.length - 1;
    int mid = 0;

    while (low <= high) {
      mid = low + (high - low) ~/ 2;
      int midValue = sortedList[mid];

      if (midValue == value) {
        return mid; // 찾은 경우, 인덱스 반환
      } else if (midValue < value) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    print(('binary search', low, mid, high));

    return high; // 값이 리스트에 없는 경우, -1 반환
  }


  void fetchInitialData() async {
    var response = await http.get(Uri.parse('$BASE_URL/pig_info/graph_page/?pNo=${widget.pNo}'));
    
    if (response.statusCode == 200) {
      var data = json.decode(response.body);

      minY = data[0]['act'];
      maxY = data[0]['act'];

      for (var idx in data) {
        date.add(DateTime.parse(idx['now']));
        act.add(idx['act'].toDouble());
        pred.add(idx['pred']);


        String? xLabelDate='';
        xLabelDate = DateFormat('yyyy-MM-dd').format(DateTime.parse(idx['now']));
        addLabelIfNotPresent(dataCount,xLabelDate);

        actData.add(FlSpot(dataCount.toDouble(), idx['act'].toDouble()));

        if (idx['pred'] != null && idx['pred'] == true) {
          ddotData_blue.add(FlSpot(dataCount.toDouble(), idx['act'].toDouble()));
          activate_num = dataCount;

          predStartTime = DateFormat('yyyy-MM-dd  HH:mm:ss').format(DateTime.parse(idx['now']));

          DateTime startTime = DateTime.parse(idx['now']);
          DateTime recommendedTime = startTime.add(Duration(hours: 28));
          predRecommendedTime = DateFormat('yyyy-MM-dd  HH:mm:ss').format(recommendedTime);
          // 디버깅을 위해 값을 출력
          //print('catching num = ${dataCount}');
        }
        if (activate_num != null) {
          int temp_num = activate_num!;
          if (dataCount == temp_num + 1680){
            ddotData_red.add(FlSpot(dataCount.toDouble(), idx['act'].toDouble()));
          }
        }
        if (act.isNotEmpty) {
          tempY = idx['act'].toDouble();
          print("tempY_for : $tempY");

          if(tempMinY >= tempY){
            tempMinY = tempY;
            minY = tempMinY-0.1;
            print("min Y: $minY");
          }
          if(tempMaxY <= tempY){
            tempMaxY = tempY;
            maxY = tempMaxY+0.2;
            print("max Y: $maxY");
          }

        } else {
          print("The 'act' list is empty.");
        }
        dataCount++;
      }
      setState(() {
        maxX = dataCount.toDouble();
      });
    } else {
      // Error handling
      print('Failed to load initial data');
    }
  }

  void setUpTimedFetch() {
    // Using a Timer to periodically fetch data
    Timer.periodic(Duration(minutes: 1), (timer) async {
      var response = await http.get(Uri.parse('$BASE_URL/pig_info/graph_page/?pNo=${widget.pNo}'));
      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        //print(data);

        if (data.length != dataCount){
          
          refreshData();
          // now continue data
          var end_data = data[dataCount];
          //print('now: ${end_data['now']}, act: ${end_data['act']}, pred: ${end_data['pred']}, act_num: ${activate_num}');

          date.add(DateTime.parse(end_data['now']));
          act.add(end_data['act'].toDouble());
          pred.add(end_data['pred']);

          String? xLabelDate='';
          xLabelDate = DateFormat('yyyy-MM-dd').format(DateTime.parse(end_data['now']));
          addLabelIfNotPresent(dataCount,xLabelDate);

          actData.add(FlSpot(dataCount.toDouble(), end_data['act'].toDouble()));

          if (end_data['pred'] != null && end_data['pred'] == true) {
            ddotData_blue.add(FlSpot(dataCount.toDouble(), end_data['act'].toDouble()));
            activate_num = dataCount;

            predStartTime = DateFormat('yyyy-MM-dd  HH:mm:ss').format(DateTime.parse(end_data['now']));

            DateTime startTime = DateTime.parse(end_data['now']);
            DateTime recommendedTime = startTime.add(Duration(hours: 28));
            predRecommendedTime = DateFormat('yyyy-MM-dd  HH:mm:ss').format(recommendedTime);
            // 디버깅을 위해 값을 출력
            //print('catching num = ${dataCount}');
          }
          if (activate_num != null) {
            int temp_num = activate_num!;
            if (dataCount == temp_num + 28){
              ddotData_red.add(FlSpot(dataCount.toDouble(), end_data['act'].toDouble()));
            }
          }

          // 데이터를 해석하고 actData 목록에 추가
          // x 값을 0부터 시작하고, y 값을 'act' 값으로 설정
          dataCount++; // 데이터 개수 증가
        }
        setState(() {
          maxX = dataCount.toDouble();
        });
      } else {
        print('Failed to load new data');
      }
    });
  }
  

  @override
  Widget build(BuildContext context){
    double screenWidth = MediaQuery.of(context).size.width;  // 화면 너비
    double screenHeight = MediaQuery.of(context).size.height; // 화면 높이

    return Scaffold(
      backgroundColor: Color(0xffA9A2C2),
      body: RefreshIndicator(
        onRefresh: ()async{
          refreshData();
        },
        child: SingleChildScrollView(
          child: Center( 
            child:Padding(
              padding: EdgeInsets.all(10.0),
              child:SizedBox(
                width: min(screenWidth * 0.9, 2000),
                height: max(screenHeight * 0.95, 1000),
                child: Container( 
                  margin: EdgeInsets.fromLTRB(0,60,0,100),
                  padding: EdgeInsets.all(30.0),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.black12, width: 2),
                    borderRadius: BorderRadius.circular(5),
                    color: Colors.white,
                  ),
                  child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Align(alignment:Alignment.centerRight,
                            child: SizedBox(
                              child: IconButton(
                              onPressed: () {
                                refreshData();
                              },
                              icon: Icon(Icons.refresh), 
                              iconSize: screenWidth < 670 ? 24.0 : 28.0,
                              color: Colors.black54,
                            ),
                          ),
                        ),
                        Text('돼지 번호 : ${widget.pNo}', 
                          style: TextStyle(
                            fontSize: screenWidth < 670 ? 17.0 : 20.0, 
                            fontWeight: FontWeight.bold
                          ),
                        ),
                        SizedBox(height:15),
                        RichText(
                          text: TextSpan(
                            children: [
                              WidgetSpan(
                                child: Container(
                                  width: 15,  // 원의 가로 크기를 조절합니다.
                                  height: 15, // 원의 세로 크기를 조절합니다.
                                  decoration: BoxDecoration(
                                    color: Colors.blue, // 빨간색으로 설정합니다.
                                    shape: BoxShape.circle, // 원 모양으로 설정합니다.
                                  ),
                                ),
                              ),
                              TextSpan(
                                text: "  발정 시작 시점 : $predStartTime",
                                style: TextStyle(fontSize: screenWidth < 670 ? 13.0 : 19.0, color: Colors.black), // 원하는 스타일로 설정합니다.
                              ),
                            ],
                          ),
                        ),
                        SizedBox(height: 5),
                        RichText(
                          text: TextSpan(
                            children: [
                              WidgetSpan(
                                child: Container(
                                  width: 15,  // 원의 가로 크기를 조절합니다.
                                  height: 15, // 원의 세로 크기를 조절합니다.
                                  decoration: BoxDecoration(
                                    color: Colors.red, // 빨간색으로 설정합니다.
                                    shape: BoxShape.circle, // 원 모양으로 설정합니다.
                                  ),
                                ),
                              ),
                              TextSpan(
                                text: "  인공수정 추천 시점 : $predRecommendedTime",
                                style: TextStyle(fontSize: screenWidth < 670 ? 13.0 : 19.0, color: Colors.black), // 원하는 스타일로 설정합니다.
                              ),
                            ],
                          ),
                        ),
                        SizedBox(height: 30),
                        Expanded(
                          child: LineChart(
                            LineChartData(
                              minX: minX,
                              maxX: maxX,
                              minY: minY,
                              maxY: maxY,
                              gridData: FlGridData(show: false),
                              titlesData: FlTitlesData(
                                leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                                rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                                topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                                bottomTitles: AxisTitles(
                                  sideTitles: SideTitles(
                                    showTitles: true,
                                    getTitlesWidget: (double value, TitleMeta meta) {
                                      // 리스트를 순회하여 값과 일치하는 좌표를 찾고 해당 라벨 위젯을 반환합니다.
                                      List<int> allKeys = labels.map((entry) => entry.key).toList();
                                      print(('allkeys',allKeys));

                                      if (value == 0) {
                                        xLabel_picked.clear();
                                        xLabel_picked.addAll(List.filled(allKeys.length, false));
                                      }

                                      int inputValue = value.toInt(); // Convert the first element of value to int
                                      // print((allKeys, inputValue));
                                      int input_key = binarySearch(allKeys, inputValue);

                                      if (input_key >= 0 && input_key < allKeys.length) {
                                        // Assuming xLabel_picked is a List<bool> and you're trying to toggle the boolean at input_key
                                        if (xLabel_picked[input_key] == false){
                                          xLabel_picked[input_key] = !xLabel_picked[input_key];
                                          int labelKey = allKeys[input_key];
                                          String labelText = labels.firstWhere((entry) => entry.key == labelKey, orElse: () => MapEntry(-1, 'Default Text')).value;
                                          return Text(labelText.substring(5));
                                        }
                                      }
                                      print(('value', value, input_key, xLabel_picked));
                                      return SizedBox.shrink();
                                    },
                                    reservedSize: 40, // 글자의 크기에 따라 조정해야 할 수 있습니다.
                                  ),
                                ),
                              ),
                              borderData: FlBorderData(
                                show: true,
                              ),
                              lineBarsData: [
                                LineChartBarData(
                                  spots: actData, // Using the actData to plot the graph
                                  isCurved: true,
                                  color: Colors.black54,
                                  barWidth: 2,
                                  belowBarData: BarAreaData(show: false),
                                  dotData: FlDotData(show:false),
                                ),
                                LineChartBarData(
                                  spots: ddotData_blue, // Using the actData to plot the graph
                                  isCurved: true,
                                  color: Colors.blue,
                                  barWidth: 2,
                                  belowBarData: BarAreaData(show: false),
                                  dotData: FlDotData(show:true),
                                ),
                                LineChartBarData(
                                  spots: ddotData_red, // Using the actData to plot the graph
                                  isCurved: true,
                                  color: Colors.red,
                                  barWidth: 2,
                                  belowBarData: BarAreaData(show: false),
                                  dotData: FlDotData(show:true,),
                                ),
                              ],
                            ),
                          ),
                        )
                      ]
                  )
                )
              ),
            ),
          )
        ),
      ),
      
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context)=> MainScreen()),
          );
        },
        child: Icon(Icons.arrow_back),
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.black,
        elevation: 0,
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.startTop,
    );
  }
}


class EmptyAppBar extends StatelessWidget implements PreferredSizeWidget{
  @override
  Widget build(BuildContext context){
    return Container();
  }
  @override
  Size get preferredSize => Size(0.0, 0.0);
}

