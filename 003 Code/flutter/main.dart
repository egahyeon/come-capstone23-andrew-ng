import 'package:flutter/material.dart';
import 'package:my_project/screen/screenLogin.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '모돈 인공수정 예측 프로그램',
      home: LoginScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}
