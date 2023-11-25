// Chakra imports
import {Flex, Text, useColorModeValue} from "@chakra-ui/react";
import Card from "components/card/Card.js";
import React, {useEffect, useState} from "react";
import Project from "views/admin/profile/components/Project";
import Menu from "../../../../components/menu/MainMenu";

export default function Projects(props) {
  const {regionId, projects, setSelectedProject} = props
  // Chakra Color Mode
  const textColorPrimary = useColorModeValue("secondaryGray.900", "white");
  const textColorSecondary = "gray.400";
  const cardShadow = useColorModeValue(
    "0px 18px 40px rgba(112, 144, 176, 0.12)",
    "unset"
  );
  const formatAsMoney = (number) => {
    const formattedNumber = Number(number).toFixed(2); // Rounds to 2 decimal places
    return formattedNumber.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  };

  return (
    <Card mb={{ base: "0px", "2xl": "20px" }}>
    <Flex alignItems="center">
      <Text
        color={textColorPrimary}
        fontWeight='bold'
        fontSize='2xl'>
        Proiectele anului 2023
      </Text>
        <Menu ms='auto' />
    </Flex>
      <Text marginBottom="20px"
          color={textColorPrimary}
      >
        Total investit: {formatAsMoney(26000000)} MDL
      </Text>
      {projects && projects.slice(0,5).map((p, i) => {
        return (
          <Project
              boxShadow={cardShadow}
              mb='20px'
              ranking='1'
              link='#'
              category={p.category_name}
              title={p.project_name}
              cursor="pointer"
              _hover={{ opacity: "0.8" }}
              onClick={() => setSelectedProject(i)}
          />
        )
      })}
    </Card>
  );
}
