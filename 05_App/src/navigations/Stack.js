import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { Home, Categories, Detail, Main, Chat } from '../screens';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const Stack = createStackNavigator();

const StackNav = () => {
    return (
        <Stack.Navigator
            initialRouteName="Home"
            screenOptions={{
                cardStyle: {backgroundColor: '#E8E1DB'},
                headerStyle: {
                    height: 100,
                    backgroundColor: '#C7B9AD',
                    borderBottomWidth: 1,
                    borderBottomColor: '#ffffff',
                    color: '#ffffff',
                },
                headerTitleStyle: {
                    fontSize: 20,
                },
                headerTitleAlign: 'center',
                headerTitle: 'Home',
                headerBackTitleVisible: false,
                headerBackImage: () =>{
                    return (
                        <MaterialCommunityIcons
                            name="arrow-left"
                            size={26}
                            color={'#000000'}
                            style={{ marginRight: 5, marginLeft: 20 }}
                        />
                    );
                },
            }}
        >
            <Stack.Screen
                name="Home"
                component={Home}
                options={{ headerShown: false }}
            />
            <Stack.Screen
                name="Categories"
                component={Categories}
                options={{ headerTitle: '카테고리' }}
            />
            <Stack.Screen
                name="Detail"
                component={Detail}
                options={{ headerTitle: '카테고리'}}
            />
            <Stack.Screen
                name="Main"
                component={Main}
                options={{ headerTitle: '상품 상세'}}
            />
            <Stack.Screen
                name="Chat"
                component={Chat}
                options={{ headerTitle: '질문하기'}}
            />
        </Stack.Navigator>
    );
};

export default StackNav;