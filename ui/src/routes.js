import React from "react";

import { Icon } from "@chakra-ui/react";
import {
  MdBarChart,
  MdPerson,
  MdHome,
  MdLock,
  MdOutlineShoppingCart, MdPhoneAndroid, MdFlag,
} from "react-icons/md";

// Admin Imports
import MainDashboard from "views/admin/default";
import Profile from "views/admin/profile";
import RTL from "views/admin/rtl";

// Auth Imports
import Region from "./views/region/Region";

const routes = [
  {
    name: "Hartă Interactivă",
    layout: "/admin",
    path: "/default",
    icon: <Icon as={MdHome} width='20px' height='20px' color='inherit' />,
    component: MainDashboard,
  },
  // {
  //   name: "Regiune",
  //   layout: "/admin",
  //   path: "/regions/:regionId",
  //   defaultPath: "/regions/2",
  //   icon: <Icon as={MdFlag} width='20px' height='20px' color='inherit' />,
  //   component: Region,
  // },
  // {
  //   name: "Profil",
  //   layout: "/admin",
  //   path: "/profile",
  //   icon: <Icon as={MdPerson} width='20px' height='20px' color='inherit' />,
  //   component: Profile,
  // },
  // {
  //   name: "Admin",
  //   layout: "/rtl",
  //   path: "/rtl-default",
  //   icon: <Icon as={MdHome} width='20px' height='20px' color='inherit' />,
  //   component: RTL,
  // },
];

export default routes;
