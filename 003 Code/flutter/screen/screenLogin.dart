//screenLogin 원본
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:my_project/screen/screenMain.dart';
import 'package:my_project/screen/screenSignup.dart';
import 'dart:convert';
import 'dart:math';

import 'package:my_project/connect_django.dart';

void main() => runApp(MaterialApp(
  title: '로그인 페이지',
  home: LoginScreen(),
));


class LoginScreen extends StatefulWidget {
  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {

  TextEditingController _userIdController = TextEditingController();
  TextEditingController _passwordController = TextEditingController();

  Future<String> login() async {
    String userId = _userIdController.text;
    String password = _passwordController.text;

    if (userId.isEmpty || password.isEmpty) {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Login Result'),
            content: Text('Please enter both user ID and password'),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.of(context).pop();
                },
                child: Text('OK'),
              ),
            ],
          );
        },
      );

      return 'success';
    }

    // final url = Uri.parse('http://127.0.0.1:8000/login_auth/login/');
    final url = Uri.parse('$BASE_URL/login_auth/login/');
    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'user_id': userId,
        'user_pw': password,
      }),
    );
    print('응답 상태 코드: ${response.statusCode}');
    print('응답 본문: ${response.body}');

    if (response.statusCode == 200) {
      return 'success';
    } else if (response.statusCode == 401) {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Login Result'),
            content: Text('Incorrect password'),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.of(context).pop();
                },
                child: Text('OK'),
              ),
            ],
          );
        },
      );
      return 'incorrect_password';
    } else {
      throw Exception('로그인 오류 발생');
    }
  }

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;
    double screenHeight = MediaQuery.of(context).size.height; // 화면 높이

    return Scaffold(
      backgroundColor: Color(0xffA9A2C2),
      body: SingleChildScrollView(
          child : Center(
            child: Padding(
              padding: EdgeInsets.all(10.0),
              child: SizedBox(
                width: min(screenWidth * 0.9, 670), 
                child: Container(
                  margin: EdgeInsets.fromLTRB(0,100,0,100),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.black12, width: 2),
                    borderRadius: BorderRadius.circular(5),
                    color : Colors.white,
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      SizedBox(height: 70),
                      Text('로그인', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                      SizedBox(height: 50),
                      SizedBox(
                        width: min(screenWidth * 0.7, 400),
                        child: TextFormField(
                          controller: _userIdController,
                          decoration: InputDecoration(
                            border: OutlineInputBorder(),
                            hintText: '아이디',
                          ),
                        ),
                      ),
                      SizedBox(height: 15),
                      SizedBox(
                        width: min(screenWidth * 0.7, 400),
                        child: TextFormField(
                          controller: _passwordController,
                          decoration: InputDecoration(
                            border: OutlineInputBorder(),
                            hintText: '비밀번호',
                          ),
                          obscureText: true,
                        ),
                      ),

                      SizedBox(height: 45),
                      ElevatedButton(
                        child: Text('로그인'),
                        style: ElevatedButton.styleFrom(
                          primary: Color(0xff6A679E),
                          onPrimary: Colors.white,
                          padding: EdgeInsets.symmetric(horizontal: 30, vertical: 18),
                          minimumSize: Size(min(screenWidth * 0.7, 400), 60),
                        ),
                        onPressed: () {
                          if (_userIdController.text.isEmpty || _passwordController.text.isEmpty) {
                            showDialog(
                              context: context,
                              builder: (BuildContext context) {
                                return AlertDialog(
                                  title: Text('로그인 실패'),
                                  content: Text('올바른 ID 또는 Password가 아닙니다.'),
                                  actions: [
                                    TextButton(
                                      onPressed: () {
                                        Navigator.of(context).pop();
                                      },
                                      child: Text('OK'),
                                    ),
                                  ],
                                );
                              },
                            );
                          } else {
                            login().then((result) {
                              if(result=='success') {
                                showDialog(
                                  context: context,
                                  builder: (BuildContext context) {
                                    return AlertDialog(
                                      title: Text('로그인 완료'),
                                      content: Text('환영합니다!'),
                                      actions: [
                                        TextButton(
                                          onPressed: () {
                                            Navigator.of(context).pop();
                                            Navigator.pushReplacement(
                                              context,
                                              MaterialPageRoute(
                                                  builder: (context) =>
                                                      MainScreen()),
                                            );
                                          },
                                          child: Text('OK'),
                                        ),
                                      ],
                                    );
                                  },
                                );
                              }
                            }).catchError((error) {
                              showDialog(
                                context: context,
                                builder: (BuildContext context) {
                                  return AlertDialog(
                                    title: Text('로그인 실패'),
                                    content: Text('Login error: $error'),
                                    actions: [
                                      TextButton(
                                        onPressed: () {
                                          Navigator.of(context).pop();
                                        },
                                        child: Text('OK'),
                                      ),
                                    ],
                                  );
                                },
                              );
                            });
                          }
                        },
                      ),
                      SizedBox(height: 25),
                      TextButton(
                        onPressed: (){
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context)=> SignupScreen()),
                          );
                        },
                        child: Text('회원가입', style: TextStyle(fontSize: 15))
                      ),
                      SizedBox(height: 70),
                    ],
                  ),
                )
              ),
            )
          )
      ),
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

