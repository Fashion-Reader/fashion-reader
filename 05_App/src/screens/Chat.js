import React, { useState, useCallback, useEffect } from "react";
import { View, Text, Dimensions } from "react-native";
import { GiftedChat } from "react-native-gifted-chat";
import { useFetch } from '../hooks/useFetch';
import axios from "axios";
import { render } from "react-dom";
import aws_url from "../others/aws_info";

const { height, width } = Dimensions.get("window");

function Chat({route, navigation}) {
  var item_id = route['params']
  // console.log("~!~!!~!",item_id,"~!~!!~!~")
  const [messages, setMessages] = useState([]);
  const question_api = aws_url+'/prod/chat/'+item_id+'/'

  useEffect(() => {
    setMessages([
      {
        _id: 1,
        text: "옷에 대해 질문을 해주세요",
        createdAt: new Date(),
        user: {
          _id: 2,
          name: "React Native",
          avatar: "https://placeimg.com/140/140/any",
        },
      },
    ]);
  }, []);

  const give_answer = async (question, len) => {
    const URL = aws_url+'/prod/chat/'+item_id+'/'+question[0].text
    console.log('^^',URL,'^^')
    var api_text;

    console.log("들어가나?");
    
    await axios.get(`${URL}`)
      .then(function (response) {
        console.log(response);
        api_text = response.data.response
      })
      .catch(function (error) {
        console.log(error);
      });

    return {
      _id: len,
      text: api_text,
      createdAt: new Date(),
      user: {
        _id: 2,
        name: "React Native",
        avatar: "https://placeimg.com/140/140/any",
      }
    }
  }

  const onSend = async (messages) => {
    console.log('1')
    setMessages((previousMessages) =>
      GiftedChat.append(previousMessages, messages)
    ); // 내 쪽 채팅
    console.log('2')
    const result = await give_answer(messages, 2)
    console.log(result, '리저트')
    setTimeout(() => setMessages((previousMessages) =>
    GiftedChat.append(
      previousMessages,
      result
    ),
  ), 8000)
    
    console.log('3')// 상대쪽 채팅
  };

  return (
    <View style={{ flex: 1 }}>
      <View
        style={{
          alignItems: "center",
          justifyContent: "center",
          position: "absolute",
          width: width,
          top: 50,
          zIndex: 10,
        }}
      >
        <Text style={{ fontSize: 30, fontWeight: "900" }}>Fashion Reader</Text>
      </View>
      <GiftedChat
        placeholder={"메세지를 입력하세요..."}
        alwaysShowSend={true}
        messages={messages}
        textInputProps={{
          keyboardAppearance: "dark",
          autoCorrect: false,
          autoCapitalize: "none",
        }}
        onSend={(messages) => onSend(messages)}
        user={{
          _id: 1,
        }}
      />
    </View>
  );
  
}

export default Chat;