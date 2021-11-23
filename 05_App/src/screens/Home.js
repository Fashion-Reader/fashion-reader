import React from 'react';
import styled from 'styled-components/native';
import Button from '../components/Button';
import {useSafeAreaInsets} from 'react-native-safe-area-context';
import { useFetch } from '../hooks/useFetch';
import aws_url from '../others/aws_info';

const Container = styled.View`
    align-items: center;
    /* background-color: #ffffff; */
    padding-top: ${({insets: {top}}) => top}px;
    padding-bottom: ${({insets: {bottom}}) => bottom}px;
    padding-right: ${({insets: {right}}) => right}px;
    padding-left: ${({insets: {left}}) => left}px;
`;

const StyledText = styled.Text`
    font-size: 30px;
    margin: 10px;
    `;



const Home = ({ navigation }) => {
    const URL = aws_url+'/prod/type/846';
    const {data, error, inProgress} = useFetch(URL);
    
    try{ // 데이터가 null로 들어오는 순간이 종종 있어서 try, catch로 에러 방지했습니다.
        // console.log("-------------------------------\n",data[0],"\n-------------------------------")
    }
    catch(e){
        // console.log("-------------------------------\n",data,"\n-------------------------------")
    }

    const insets = useSafeAreaInsets();
    // console.log(insets);
    return (
        <Container insets={insets}>
            <StyledText>홈</StyledText>
            <Button title="임블리" onPress={() => navigation.navigate('Categories')} />
        </Container>
    );
};

export default Home;