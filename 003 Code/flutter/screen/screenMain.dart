import 'dart:async';
import 'dart:convert';
import 'package:intl/intl.dart';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:my_project/screen/screenDetail.dart';
import 'package:my_project/screen/screenLogin.dart';
import 'dart:math';


import 'package:my_project/connect_django.dart';

// 데이터를 담을 클래스를 정의합니다.
class PigInfo {
  final int pNo;
  final String now;

  PigInfo({required this.pNo, required this.now});

  factory PigInfo.fromJson(Map<String, dynamic> json) {
    // 'now' 필드를 읽고 null인 경우 기본값으로 설정합니다.
    final String? nowString = json['now'];
    String parsedNow;

    // nowString이 null이 아니면 분할 로직을 적용합니다.
    if (nowString != null) {
      final List<String> splitDateTime = nowString.split(RegExp(r"[T\+]"));
      parsedNow = '${splitDateTime[0]} ${splitDateTime[1]}';
    } else {
      // nowString이 null일 경우 기본 문자열을 설정합니다.
      parsedNow = '예측 발정 전';
    }

    return PigInfo(
      pNo: json['pNo'],
      now: parsedNow, // 'now' 필드를 문자열로 변경
    );
  }
}

class MainScreen extends StatefulWidget{
  @override
  _MainScreenState createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  late Future<List<PigInfo>> pigInfoList;

  @override
  void initState() {
    super.initState();
    pigInfoList = fetchPigInfo();
  }

  Future<List<PigInfo>> fetchPigInfo() async {
    final response = await http.get(Uri.parse('$BASE_URL/pig_info/main_page/'));

    if (response.statusCode == 200) {
      List jsonResponse = json.decode(response.body);
      return jsonResponse.map((data) => PigInfo.fromJson(data)).toList();
    } else {
      throw Exception('Failed to load pig info from the server');
    }
  }

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;
    double screenHeight = MediaQuery.of(context).size.height; 

    return Scaffold(
      backgroundColor: Color(0xffA9A2C2),
      body: SingleChildScrollView(
        child: Center(
          child:Padding(
            padding: EdgeInsets.all(10.0),
            child:SizedBox(
              width: min(screenWidth * 0.9, 850), 
              child: Container(
                margin: EdgeInsets.fromLTRB(0,100,0,100),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.black12, width: 2),
                  borderRadius: BorderRadius.circular(5),
                  color: Colors.white,
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.start,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    SizedBox(height: 70),
                    Text('최근 인공수정 적기 순', 
                      style: TextStyle(
                        fontSize: screenWidth < 730 ? 16.0 : 19.0, 
                        fontWeight: FontWeight.bold
                      ),
                    ),
                    SizedBox(height: 50),
                    Padding(
                      padding: EdgeInsets.symmetric(horizontal: 30),
                      child: FutureBuilder<List<PigInfo>>(
                        future: pigInfoList,
                        builder: (context, snapshot) {
                          if (snapshot.connectionState == ConnectionState.waiting) {
                            return Center(child: CircularProgressIndicator());
                          } else if (snapshot.hasError) {
                            return Center(child: Text('Error: ${snapshot.error}'));
                          } else {
                            // Filter the list to only include the latest entry for each pNo
                            Map<int, PigInfo> latestPigInfo = {};
                            for (var pig in snapshot.data!) {
                              latestPigInfo[pig.pNo] = pig;
                            }
                            List<PigInfo> latestPigList = latestPigInfo.values.toList();

                            // Sort the list by datetime in descending order
                            latestPigList.sort((a, b) {
                              return b.now.compareTo(a.now); // 'now' 필드를 문자열로 변경했으므로 compareTo를 사용합니다.
                            });
                            return ListView.separated(
                              shrinkWrap: true,
                              itemCount: latestPigList.length,
                              itemBuilder: (context, index) {
                                final item = latestPigList[index];
                                return ListTile(
                                  title: Text(
                                    '돼지 번호 : ${item.pNo}',
                                    style: TextStyle(fontSize: screenWidth < 730 ? 12.0 : 16.0),
                                  ),
                                  subtitle: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        '인공수정 적기',
                                        style: TextStyle(fontSize: MediaQuery.of(context).size.width < 730 ? 12.0 : 15.0,),
                                      ),
                                      Text(
                                        '${item.now}',
                                        style: TextStyle(fontSize: MediaQuery.of(context).size.width < 730 ? 12.0 : 15.0,),
                                      ),
                                    ],
                                  ),
                                  trailing: ElevatedButton(
                                    onPressed: () {
                                      Navigator.push(
                                        context,
                                        MaterialPageRoute(
                                          builder: (context) => DetailScreen(pNo: item.pNo),
                                        ),
                                      );
                                    },
                                    child: Text(
                                      '상세보기',
                                      style: TextStyle(
                                        fontSize: screenWidth < 730 ? 11.0 : 15.0, // Adjust the font size here
                                      ),
                                    ),
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: Colors.deepPurple,
                                      minimumSize: Size(screenWidth < 730 ? 50 : 100, screenWidth < 730 ? 36.0 : 40.0),
                                    ),
                                  ),
                                );
                              },
                              separatorBuilder: (context, index) {
                                // 각 항목 사이에 구분선을 추가합니다.
                                return Divider(thickness: 1);
                              },
                            );
                          }
                        },
                      ),
                    ),
                    SizedBox(height: 200),
                  ],
                ),
              ),

            ),
          ),
          
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context)=> LoginScreen()),
          );
        },
        child: Icon(Icons.logout_outlined),
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.black,
        elevation: 0,
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.endTop,
    );
  }
}