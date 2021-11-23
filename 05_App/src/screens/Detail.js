import React from 'react';
import styled from 'styled-components/native';
import DetailButton from '../components/DetailButton';
import {useSafeAreaInsets} from 'react-native-safe-area-context';
import { useFetch } from '../hooks/useFetch';
import aws_url from '../others/aws_info';

const Container = styled.ScrollView`
    /* align-items: center; */
    /* flex-direction: row; */
    /* background-color: #ffffff; */
    padding-top: 10px;
    padding-bottom: 100px;
    padding-right: ${({insets: {right}}) => right}px;
    padding-left: ${({insets: {left}}) => left}px;
    align-self: center;
    margin-bottom: 100px;
`;

const StyledText = styled.Text`
    font-size: 30px;
    margin: 10px;
    `;

const Detail = ({ route, navigation }) => {
    const insets = useSafeAreaInsets();
    console.log('^^^^^^^^^^^^^',aws_url);
    var category_id = route['params']
    // console.log("$#$#$#$#",category_id,"@#@$#$@#$")
    var loop = [];
    const URL = aws_url+'/prod/type/'+category_id;
    console.log("@@@@@@@@@@",URL,"@@@@@@@@@@")
    const {data, error, inProgress} = useFetch(URL);

    try{ // 데이터가 null로 들어오는 순간이 종종 있어서 try, catch로 에러 방지했습니다.
        // console.log("-----------**--------------------\n",data,"\n---------------**----------------")
        // console.log('&%&%&',data[0])
        for (let i=0; i < data[0].length; i++){
            loop.push(
                <DetailButton title={data[0][i]['name']} onPress={() => navigation.navigate('Main',data[0][i]['item_id'])} uri={data[0][i]['item_img_links']} />
            );
        }
    }
    catch(e){
        // console.log("-------------------------------\n",data,"\n-------------------------------")
    }

    return (
        <Container insets={insets}>
            {loop}
        </Container>
    );
};

export default Detail;