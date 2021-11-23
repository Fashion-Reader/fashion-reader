import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, Text, Touchable, TouchableOpacity, View, Image } from 'react-native';
import styled from 'styled-components/native';
import { icons } from './icons';
import { sample } from './sample';
import CustomButton from './CustomButton';
import { useFetch } from './hooks/useFetch';

const Container = styled.ScrollView`
    flex:1;
`;

const Icon = styled.Image`
    width: 30px;
    height: 30px;
    /* margin: 10px; */
`;

const SampleImage = styled.Image`
    flex:1;
    width: auto;
`;

const styles = StyleSheet.create({
    background: {
        flex:1,
        backgroundColor:"white",
    },
    header: {
        // flex:1,
        height: 100,
        flexDirection:'row',
        alignItems:"center",
        justifyContent:"space-between",
        backgroundColor: "white",
    },
    main: {
        flex:8,
        backgroundColor:"#C7B9AD",
    },
    topleft: {
        marginLeft: 30,
        width:50,
        height:50,
        top:40,
    },
    topmid: {
        width:80,
        height:50,
        top:40,
        fontWeight:'bold'
    },
    topright: {
        width:50,
        height:50,
        top: 40,
    },
    title: {
        fontSize:15,
    },
    imageView: {
        flex:1.4,
        backgroundColor: 'white',
        borderColor: 'gray',
        borderWidth:1,
    },
    detailView: {
        flex:1,
        backgroundColor: 'white',
        borderColor: 'gray',
        borderWidth:1,
    },
});

const RESTAPI = ({something}) => {
    return (
        <Text style={styles.title,{fontSize:20, marginLeft:20}}>
            {something}
        </Text>
    );
};

export default function App() {
    const URL = 'http://15.165.189.139:8000/api/tutorials';
    const {data, error, inProgress} = useFetch(URL);
    console.log(data);
    
    return (
        <Container>
        {/* <View style={styles.background}> */}
            <View style={styles.header}>
                <TouchableOpacity style={styles.topleft} onPress={() => alert("빠꾸")}>
                    <View>
                        <Icon source = { icons.back }></Icon>
                    </View>
                </TouchableOpacity>
                <Text style={styles.topmid}>상품 상세</Text>
                <TouchableOpacity style={styles.topright} onPress={() => alert("홈")}>
                    <View>
                        <Icon source = { icons.home }></Icon>
                    </View>
                </TouchableOpacity>
            </View>
            <View style={styles.main}>
                <View style={styles.imageView} >
                    {/* <SampleImage source = { sample.hood }></SampleImage> */}
                    {data?.map(({name, price, url}) => (
                        <Image
                            style={{width:400, height:400}}
                            source = {{uri: url}}/>
                    ))}
                </View>
                <View style={styles.detailView}>
                    <Text style={styles.title,{fontSize:15, marginLeft:20, marginTop:30}}>
                        {/* 마크 곤잘레스 */}
                    </Text>
                    
                    {data?.map(({ name, price, url }) => (
                        <RESTAPI key = {name} something={name} />
                    ))}
                    <Text style={styles.title,{fontSize:20, fontWeight:'bold' ,marginLeft:20, marginTop:25}}>
                    {data?.map(({ name, price, url }) => (
                        <RESTAPI key = {name} something={price} />
                    ))}
                    </Text>
                    <Text style={styles.title,{fontSize:15, marginLeft:20}}>
                        배송비 3,000원
                    </Text>
                    <View style={styles.header,{alignItems:'flex-start'}}>
                        <View style={styles.header,{flexDirection:'row'}}>
                            <CustomButton title="질문하기" onPress={() => alert("질문하기")}/>
                            <CustomButton title="리뷰 요약" onPress={() => alert("리뷰 요약")}/>
                        </View>
                        <View style={styles.header,{flexDirection:'row'}}>
                            <CustomButton title="상세정보" onPress={() => alert("상세정보")}/>
                            <CustomButton title="상품문의" onPress={() => alert("상품문의")}/>
                        </View>
                        <View style={styles.header,{flexDirection:'row'}}>
                            <CustomButton title="필수표기정보" onPress={() => alert("필수표기정보")}/>
                            <CustomButton title="배송/교환/환불 안내" onPress={() => alert("배송/교환/환불 안내")}/>
                        </View>
                        
                    </View>
                </View>
            </View>
        {/* </View> */}
        </Container>
    );
}