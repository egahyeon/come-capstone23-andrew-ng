//screenSignup.dart 원본

import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:my_project/screen/screenLogin.dart';
import 'dart:math';

import 'package:my_project/connect_django.dart';

class SignupScreen extends StatefulWidget {
  @override
  _SignupScreenState createState() => _SignupScreenState();
}

class _SignupScreenState extends State<SignupScreen> {

  TextEditingController _userIdController = TextEditingController();
  TextEditingController _emailController = TextEditingController();
  TextEditingController _passwordController = TextEditingController();
  TextEditingController _confirmPasswordController = TextEditingController();

  Future<void> signup() async {
    final url = Uri.parse('$BASE_URL/login_auth/signup/');
    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'user_id': _userIdController.text,
        'user_pw': _passwordController.text,
        'user_email': _emailController.text,
      }),
    );

    if (response.statusCode == 201) {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Signup Result'),
            content: Text('Signup successful'),
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
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Signup Result'),
            content: Text('Signup failed'),
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
    }
  }

  bool isAllFieldsFilled() {
    return _userIdController.text.isNotEmpty &&
        _emailController.text.isNotEmpty &&
        _passwordController.text.isNotEmpty &&
        _confirmPasswordController.text.isNotEmpty;
  }

  bool isEmailValid(String email) {
    // 정규 표현식을 사용하여 이메일 형식을 검사합니다.
    final emailRegex = RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$');
    return emailRegex.hasMatch(email);
  }

  bool doPasswordsMatch() {
    return _passwordController.text == _confirmPasswordController.text;
  }

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;
    double screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
      backgroundColor: Color(0xffA9A2C2),
      body: SingleChildScrollView(
          child : Center(
              child: Padding(
                padding: EdgeInsets.all(10.0),
                child:SizedBox(
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
                        Text('회원가입', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                        SizedBox(height: 30),
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
                            controller: _emailController,
                            decoration: InputDecoration(
                              border: OutlineInputBorder(),
                              hintText: '이메일',
                            ),
                            keyboardType: TextInputType.emailAddress,
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
                        SizedBox(height: 15),
                        SizedBox(
                          width: min(screenWidth * 0.7, 400),
                          child: TextFormField(
                            controller: _confirmPasswordController,
                            decoration: InputDecoration(
                              border: OutlineInputBorder(),
                              hintText: '비밀번호 재확인',
                            ),
                            obscureText: true,
                          ),
                        ),
                        SizedBox(height: 45),
                        ElevatedButton(
                          child: Text(
                            '회원가입하기',
                            style: TextStyle(fontSize: 16.0),
                          ),
                          style: ElevatedButton.styleFrom(
                            primary: Color(0xff6A679E),
                            onPrimary: Colors.white,
                            padding: EdgeInsets.symmetric(horizontal: 30, vertical: 18),
                            minimumSize: Size(min(screenWidth * 0.7, 400), 60),
                          ),
                          onPressed: () {
                            if (isAllFieldsFilled() && doPasswordsMatch() && isEmailValid(_emailController.text)) {
                              signup().then((result) {
                                showDialog(
                                  context: context,
                                  builder: (BuildContext context) {
                                    return AlertDialog(
                                      title: Text('Signup Result'),
                                      content: Text('회원가입이 완료되었습니다.'),
                                      actions: [
                                        TextButton(
                                          onPressed: () {
                                            Navigator.push(
                                              context,
                                              MaterialPageRoute(
                                                  builder: (context) => LoginScreen()),
                                            );
                                          },
                                          child: Text('OK'),
                                        ),
                                      ],
                                    );
                                  },
                                );
                              });
                            } else {
                              // Show an error message if the conditions are not met
                              String errorMessage = '모든 필드를 작성하고 유효한 이메일을 입력하세요.';
                              if (!isAllFieldsFilled()) {
                                errorMessage = '모든 필드를 작성해주세요.';
                              } else if (!doPasswordsMatch()) {
                                errorMessage = '비밀번호가 일치하지 않습니다.';
                              } else if (!isEmailValid(_emailController.text)) {
                                errorMessage = '유효한 이메일을 입력하세요.';
                              }

                              showDialog(
                                context: context,
                                builder: (BuildContext context) {
                                  return AlertDialog(
                                    title: Text('Signup Result'),
                                    content: Text(errorMessage),
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
                            }
                          },
                        ),
                        SizedBox(height: 20),
                        ElevatedButton(
                          child: Text('로그인하기', style: TextStyle(fontSize: 15)),
                          style: ElevatedButton.styleFrom(
                            primary: Color(0xff6A679E),
                            onPrimary: Colors.white,
                            padding: EdgeInsets.symmetric(horizontal: 30, vertical: 18),
                            minimumSize: Size(min(screenWidth * 0.7, 400),60),
                          ),
                          onPressed: (){
                            Navigator.push(
                              context,
                              MaterialPageRoute(builder: (context)=> LoginScreen()),
                            );
                          },
                        ),
                        SizedBox(height: 70),
                      ],
                    ),
                  )
                ),
              ),
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

