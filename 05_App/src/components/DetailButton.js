import React from 'react';
import { TouchableOpacity, Image } from 'react-native';
import styled from 'styled-components/native';

const Container = styled.View`
    background-color: #F4F0ED;
    padding: 10px;
    margin: 10px;
    width: 350;
    height: 80;
    border-radius: 15;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    /* align-self: flex-start; */
`;

const Title = styled.Text`
    font-size: 18px;
    color: #000000;
`;

const DetailButton = ({ title, onPress, uri }) => {
    return (
        <TouchableOpacity onPress={onPress}>
            <Container>
                <Image
                    style={{width:50, height:50}}
                    source = {{uri: uri}}
                    borderRadius = {30}
                    />
                <Title>{title}</Title>
            </Container>
        </TouchableOpacity>
    );
};

export default DetailButton;
